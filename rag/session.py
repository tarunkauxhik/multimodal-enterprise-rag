"""Session-scoped ingestion and question answering for the Streamlit app.

Two collection models, deliberately kept separate:
- CLI (`python -m rag.ingest`) writes to the shared QDRANT_COLLECTION ("documents").
- The Streamlit app uses one private collection per browser session, created here.
  The app never reads the shared collection, so CLI-ingested documents never
  appear in a session.

Isolation: the session collection name is generated on the server and kept only
in that session's state; ingestion, BM25 and dense search all run against it.

Lifetime: each session collection stores its last activity time in Qdrant
collection metadata. Uploads, questions and ongoing app use refresh it, and
collections inactive for SESSION_TTL_SECONDS are deleted whenever a new session
starts. A browser refresh starts a new Streamlit session: the old collection stops
being refreshed and is removed by the same inactivity cleanup.

Threads: Streamlit runs each session on its own thread. The API clients in
Services (httpx-based Jina and MiniMax clients, google-genai, qdrant-client) are
shared across sessions on the assumption that they are safe for concurrent
requests, which is their normal usage; this is not load-tested yet. SQLite caches
are not shared: they are opened per operation.
"""

import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import (
    EMBED_CACHE_PATH,
    EMBED_TASK_DOCUMENT,
    EMBED_TASK_QUERY,
    GENERATION_THINKING,
    UNDERSTAND_CACHE_PATH,
    UNDERSTAND_MAX_TOKENS,
    load_settings,
)
from rag.embed import EmbedBatch, EmbeddingCache, gemini_embedder
from rag.generate import Answer, Complete, generate_answer, minimax_client
from rag.ingest import IngestReport, IngestResult, ingest_pdf, report_stage
from rag.rerank import Rerank, jina_reranker
from rag.retrieve import Hit, Retriever
from rag.store import ensure_collection
from rag.understand import UnderstandingCache

SESSION_PREFIX = "session_"
SESSION_TTL_SECONDS = 24 * 60 * 60  # inactivity before a session's documents are deleted
ACTIVITY_KEY = "last_activity"  # Qdrant collection metadata key, unix seconds


@dataclass
class Services:
    """API clients shared by all sessions (see the thread-safety note in the module docstring)."""

    client: QdrantClient
    embed_document: EmbedBatch
    embed_query: EmbedBatch
    rerank: Rerank
    answer_model: Complete
    understand_model: Complete
    embed_cache_path: Path = EMBED_CACHE_PATH
    understanding_cache_path: Path = UNDERSTAND_CACHE_PATH
    secrets: tuple[str, ...] = ()  # redacted from any error shown to users


def build_services() -> Services:
    s = load_settings()
    return Services(
        client=QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key),
        embed_document=gemini_embedder(s.gemini_api_key, EMBED_TASK_DOCUMENT),
        embed_query=gemini_embedder(s.gemini_api_key, EMBED_TASK_QUERY),
        rerank=jina_reranker(s.jina_api_key),
        answer_model=minimax_client(s.minimax_api_key, s.minimax_base_url, thinking=GENERATION_THINKING),
        understand_model=minimax_client(s.minimax_api_key, s.minimax_base_url, max_tokens=UNDERSTAND_MAX_TOKENS),
        secrets=tuple(k for k in (s.minimax_api_key, s.gemini_api_key, s.jina_api_key, s.qdrant_api_key) if k),
    )


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def new_session_collection(now: float | None = None) -> str:
    return f"{SESSION_PREFIX}{int(_now(now))}_{uuid.uuid4().hex}"


def start_session(client: QdrantClient, now: float | None = None) -> str:
    """Create a private collection for a new browser session and record its first activity."""
    collection = new_session_collection(now)
    ensure_collection(client, collection)
    touch_session(client, collection, now)
    return collection


def touch_session(client: QdrantClient, collection: str, now: float | None = None) -> bool:
    """Record activity on a session collection. Returns False if it no longer exists (expired or cleared)."""
    if not client.collection_exists(collection):
        return False
    client.update_collection(collection, metadata={ACTIVITY_KEY: int(_now(now))})
    return True


def last_activity(client: QdrantClient, collection: str) -> int | None:
    value = (client.get_collection(collection).config.metadata or {}).get(ACTIVITY_KEY)
    if isinstance(value, (int, float)):
        return int(value)
    try:  # no activity recorded (e.g. created by older code): use the creation time in the name
        return int(collection[len(SESSION_PREFIX) :].split("_", 1)[0])
    except ValueError:
        return None


def cleanup_inactive_sessions(
    client: QdrantClient, now: float | None = None, ttl: int = SESSION_TTL_SECONDS
) -> list[str]:
    """Delete session collections with no activity for more than `ttl`. Other collections are never touched."""
    now = _now(now)
    deleted = []
    for collection in client.get_collections().collections:
        name = collection.name
        if not name.startswith(SESSION_PREFIX):
            continue
        try:
            active = last_activity(client, name)
            if active is not None and now - active > ttl:
                client.delete_collection(name)
                deleted.append(name)
        except Exception:  # e.g. already deleted by another session's cleanup; never block session start
            continue
    return deleted


def delete_session(client: QdrantClient, collection: str) -> None:
    client.delete_collection(collection)


def unique_source_name(name: str, taken: Iterable[str]) -> str:
    """Keep file names unique within a session so [document, Page N] citations are unambiguous."""
    taken = set(taken)
    if name not in taken:
        return name
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    n = 2
    while (candidate := f"{stem} ({n}){dot}{ext}") in taken:
        n += 1
    return candidate


def upload_size_error(size_bytes: int, limit_mb: int) -> str | None:
    if size_bytes > limit_mb * 1024 * 1024:
        return (
            f"File is {size_bytes / (1024 * 1024):.1f} MB, above the {limit_mb} MB upload limit "
            "(server.maxUploadSize / STREAMLIT_SERVER_MAX_UPLOAD_SIZE)."
        )
    return None


def ingest_upload(
    services: Services,
    collection: str,
    data: bytes,
    source_name: str,
    on_stage: Callable[[str], None] | None = None,
    report: IngestReport | None = None,
) -> IngestResult:
    # Failures around ingest_pdf are reported as "setup" and "finish", never as extract/done.
    if report is None:
        report = IngestReport(source_name)
    else:
        report.reset(source_name)
    with report_stage(report, "setup"):
        embed_cache = EmbeddingCache(services.embed_cache_path)
        understanding_cache = UnderstandingCache(services.understanding_cache_path)
    try:
        result = ingest_pdf(
            data,
            source_name,
            client=services.client,
            embed_batch=services.embed_document,
            cache=embed_cache,
            complete=services.understand_model,
            understanding_cache=understanding_cache,
            collection=collection,
            on_stage=on_stage,
            report=report,
        )
    finally:
        embed_cache.close()
        understanding_cache.close()
    with report_stage(report, "finish"):  # the document is already stored when this runs
        touch_session(services.client, collection)  # after ingestion, which may take minutes
    report.stage = "done"
    return result


def answer_question(services: Services, collection: str, question: str) -> tuple[Answer, list[Hit]]:
    embed_cache = EmbeddingCache(services.embed_cache_path)
    try:
        retriever = Retriever(services.client, services.embed_query, embed_cache, services.rerank, collection=collection)
        hits = retriever.retrieve(question)
    finally:
        embed_cache.close()
    touch_session(services.client, collection)
    return generate_answer(question, [h.payload for h in hits], services.answer_model), hits


def redact(message: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        message = message.replace(secret, "***")
    return message
