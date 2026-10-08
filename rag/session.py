"""Ingestion and question answering against one Qdrant collection, as used by the HTTP API (api.py).

build_services creates the API clients once; ingest_upload and answer_question run the
unchanged rag/ pipeline against whichever collection the caller names. respond is the chat entry
point: it rewrites follow-ups into standalone requests when earlier turns are given
(rag.contextualize), routes each message (rag.route) and retrieves only for document-content questions.
The API uses one shared collection (API_COLLECTION, default QDRANT_COLLECTION, the one the CLI
writes to).

Private, expiring collections (start_session, touch_session, cleanup_inactive_sessions,
delete_session): each stores its last activity time in Qdrant collection metadata and is
deleted after SESSION_TTL_SECONDS of inactivity. They served the former per-browser
Streamlit app; the API does not use them.

Threads: the API clients in Services (httpx-based Jina and LLM clients, google-genai,
qdrant-client) are shared across request and ingestion threads on the assumption that they
are safe for concurrent requests, which is their normal usage; this is not load-tested yet.
SQLite caches are not shared: they are opened per operation.
"""

import re
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient

from rag.config import (
    EMBED_CACHE_PATH,
    EMBED_TASK_DOCUMENT,
    EMBED_TASK_QUERY,
    GENERATION_REASONING_EFFORT,
    UNDERSTAND_CACHE_PATH,
    UNDERSTAND_MAX_TOKENS,
    UNDERSTAND_REASONING_EFFORT,
    REWRITE_MAX_TOKENS,
    REWRITE_REASONING_EFFORT,
    REWRITE_TIMEOUT,
    load_settings,
)
from rag.contextualize import Turn, contextualize
from rag.embed import EmbedBatch, EmbeddingCache, gemini_embedder
from rag.generate import Answer, Complete, generate_answer, llm_client
from rag.route import NO_WORKSPACE, WHAT_IS_THIS, Route, classify, expand_term, normalize
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
    rewrite_model: Complete | None = None  # follow-up rewriting (rag.contextualize); None: messages are used as typed
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
        answer_model=llm_client(s.xai_api_key, s.xai_base_url, reasoning_effort=GENERATION_REASONING_EFFORT),
        understand_model=llm_client(
            s.xai_api_key, s.xai_base_url, max_tokens=UNDERSTAND_MAX_TOKENS, reasoning_effort=UNDERSTAND_REASONING_EFFORT
        ),
        # One attempt and a short timeout: a slow rewrite falls back to the message as typed.
        rewrite_model=llm_client(
            s.xai_api_key, s.xai_base_url, max_tokens=REWRITE_MAX_TOKENS, reasoning_effort=REWRITE_REASONING_EFFORT,
            attempts=1, timeout=REWRITE_TIMEOUT,
        ),
        secrets=tuple(k for k in (s.xai_api_key, s.gemini_api_key, s.jina_api_key, s.qdrant_api_key) if k),
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


# --- chat: route first, retrieve only for document-content questions (rag.route) -----------------

ReplyKind = Literal["answer", "abstain", "out_of_scope", "conversation", "workspace", "clarify"]


@dataclass
class Reply:
    """One chat reply. Only "answer" carries document facts, and only with validated citations."""

    kind: ReplyKind
    text: str
    citations: list[tuple[str, int]] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)  # follow-up questions offered as one-tap replies


CONVERSATION = {
    "greeting": "Hey 👋 What are we digging into?",
    "thanks": "Anytime.",
    "ack": "👍",
    "farewell": "See you.",
    "help": "I answer questions about the documents in this workspace, citing the pages I use. "
    "Ask about a topic, a table or a specific page, or ask for an overview.",
}
EMPTY_WORKSPACE = "There are no documents here yet, so there's nothing to answer from. Add a PDF and ask again."
UNCLEAR = "Which part do you mean? Give me a topic, a page, or a question and I'll dig in."
OVERVIEW_CHUNKS = 8  # opening chunks of one document: title, contents, introduction
OVERVIEW_WINDOW = 30  # pages searched for them: long reports put contents after covers and abbreviation lists
OVERVIEW_MIN_CHARS = 80  # shorter chunks (cover lines, running titles) carry no overview information
CONTENTS_SECTION = re.compile(r"\b(table of )?contents?\b|विषय[\s-]*सूची|अनुक्रमणिका", re.IGNORECASE)
SKIPPED_SECTION = re.compile(r"abbreviation|acronym|glossary|list of (tables|figures|charts)|nomenclature", re.IGNORECASE)
CONTEXT_LIMIT = 12  # chunks sent for overview and page answers, about the size of a normal answer's context


def _md(name: str) -> str:
    """A document name as Markdown text: formatting characters escaped, so a name cannot format the reply."""
    return "".join("\\" + c if c in "\\`*_[]<>#|~" else c for c in name)


def _chunk_order(p: dict) -> tuple[int, int]:
    tail = str(p.get("chunk_id", "")).rsplit("-", 1)[-1]  # chunk_id = {document_id}-p{page}-{n}
    return int(p["page_number"]), int(tail) if tail.isdigit() else 0


def overview_evidence(chunks: list[dict]) -> list[dict]:
    """The chunks an overview is answered from, for one document (chunks in reading order).

    The first chunk (title), the document's own contents section, then informative early chunks:
    near-empty chunks and abbreviation/glossary lists are skipped, so a long report's covers and
    acronym tables cannot crowd out its contents page. All within the first OVERVIEW_WINDOW pages.
    """
    early = [p for p in chunks if int(p["page_number"]) <= OVERVIEW_WINDOW] or chunks

    def sections(p: dict) -> list[str]:
        return p.get("section_path") or []

    contents = [p for p in early[1:] if any(CONTENTS_SECTION.search(s) for s in sections(p))][:4]
    candidates = [p for p in early[1:] if p not in contents and not any(SKIPPED_SECTION.search(s) for s in sections(p))]
    informative = [p for p in candidates if len(p["text"].strip()) >= OVERVIEW_MIN_CHARS]
    short = [p for p in candidates if p not in informative]  # used only when slots remain (short documents)
    chosen = (early[:1] + contents + informative + short)[:OVERVIEW_CHUNKS]
    return sorted(chosen, key=_chunk_order)


def _answer_reply(answer: Answer) -> Reply:
    if not answer.abstained:
        return Reply("answer", answer.text, answer.citations, answer.sources)
    return Reply("out_of_scope" if answer.out_of_scope else "abstain", answer.text)


def _overview(name: str) -> str:
    return f"Give me an overview of {name}"


def _as_asked(typed: str, standalone: str) -> str:
    """The question the generator answers: the user's own words, plus their standalone reading when a
    follow-up was rewritten. The answer follows what was typed (format, language, scope), while the
    reading says what "it" or "the next page" refers to. Retrieval uses the reading alone."""
    return typed if standalone == typed else f"{typed}\n(In this conversation: {standalone})"


def respond(services: Services, collection: str, message: str, history: Sequence[Turn] = ()) -> Reply:
    """Answer one chat message: route it (rag.route), then use only the evidence that route needs.

    Conversation replies touch nothing. With earlier turns (`history`, oldest first), any other
    message is first rewritten into a standalone request (rag.contextualize), so "tell me more" or
    "what about page 4?" is routed and retrieved like a first message; history is never evidence.
    Metadata and clarifications read the stored chunk payloads. Overview and page questions send
    chosen chunks to the same grounded generator as retrieval, so citation validation and
    abstention apply unchanged. Everything else is the full pipeline, limited to one document when
    the (rewritten) message names it.
    """
    route = classify(message)
    if route.intent in NO_WORKSPACE:
        return Reply("conversation", CONVERSATION[route.intent])

    embed_cache = EmbeddingCache(services.embed_cache_path)
    try:
        retriever = Retriever(services.client, services.embed_query, embed_cache, services.rerank, collection=collection)
        payloads = retriever.payloads
        if not payloads:
            return Reply("abstain", EMPTY_WORKSPACE)
        last_page: dict[str, int] = {}
        for p in payloads:
            last_page[p["source_name"]] = max(last_page.get(p["source_name"], 0), int(p["page_number"]))
        names = sorted(last_page)
        typed = message
        message = contextualize(message, history, names, services.rewrite_model)  # unchanged without history
        asked = _as_asked(typed, message)  # what the generator answers
        route = classify(message, names)
        if route.intent in NO_WORKSPACE:  # the rewrite kept a social message as it was
            return Reply("conversation", CONVERSATION[route.intent])
        if route.intent == "followup":  # nothing earlier to continue from: ask, as for a vague message
            route = Route("vague", document=route.document)
        scope = [route.document] if route.document else names

        # Only pages with indexed text are known (the PDF's own page count is not stored), so replies
        # say "indexed pages" and never claim a total length.
        indexed: dict[str, set[int]] = {}
        for p in payloads:
            indexed.setdefault(p["source_name"], set()).add(int(p["page_number"]))

        def extent(name: str) -> str:
            count = len(indexed[name])
            text = f"{count} indexed {'page' if count == 1 else 'pages'}"
            return text if count == last_page[name] else f"{text}, up to page {last_page[name]}"

        if route.intent == "list_documents":
            listed = "\n".join(f"- **{_md(n)}**, {extent(n)}" for n in names[:10])
            more = f"\n- and {len(names) - 10} more" if len(names) > 10 else ""
            text = (f"You have one document: **{_md(names[0])}**, {extent(names[0])}." if len(names) == 1
                    else f"You have {len(names)} documents:\n\n{listed}{more}")
            return Reply("workspace", text, suggestions=[_overview(names[0])])
        if route.intent == "page_count":
            text = "\n".join(f"**{_md(n)}**: {extent(n)}." for n in scope[:10])
            return Reply("workspace", text)
        if route.intent == "vague":
            if len(names) == 1:
                name = _md(names[0])
                text = (f"Do you mean **{name}**? I can give you a quick overview." if WHAT_IS_THIS.search(normalize(message))
                        else f"You've got **{name}** loaded. Want a summary, a topic, or a specific page?")
            else:
                text = f"You've got {len(names)} documents loaded. Which one, and do you want a summary, a topic, or a specific page?"
            return Reply("clarify", text, suggestions=[_overview(n) for n in names[:2]])
        if route.intent == "term":
            term = route.term or ""
            # Expanded per document: the same abbreviation can mean different things in different files.
            expansions: dict[str, list[str]] = {}
            for name in scope:
                texts = [p["text"] for p in payloads if p["source_name"] == name]
                texts += [" / ".join(p.get("section_path") or []) for p in payloads if p["source_name"] == name]
                if found := expand_term(term, texts):
                    expansions.setdefault(found, []).append(name)
            if len(expansions) == 1:
                (expansion,) = expansions
                return Reply("clarify", f"Do you mean {term.upper()} ({expansion})? I can pull up the relevant section.",
                             suggestions=[f"What is {expansion}?"])
            if len(expansions) > 1:
                options = [f"**{e}** (in {_md(docs[0])})" for e, docs in list(expansions.items())[:3]]
                return Reply("clarify", f"{term.upper()} means different things in your documents: {' or '.join(options)}. Which one?",
                             suggestions=[f"What is {e}?" for e in list(expansions)[:3]])
            if not any(re.search(rf"\b{re.escape(term)}\b", p["text"], re.IGNORECASE) for p in payloads):
                return Reply("clarify", UNCLEAR)
            route = Route("question", query=f"What do the documents say about {term}?")

        if route.intent == "overview" and len(scope) > 1:
            # Several documents and none named: mixing their opening pages answers nothing well.
            return Reply("clarify", f"Which document do you mean? You've got {len(names)} loaded.",
                         suggestions=[_overview(n) for n in names[:3]])
        if route.intent == "overview":
            (name,) = scope
            chunks = sorted((p for p in payloads if p["source_name"] == name), key=_chunk_order)
            return _answer_reply(generate_answer(asked, overview_evidence(chunks), services.answer_model))
        if route.intent == "page":
            pages = [n for n in route.pages if n >= 1]
            if not pages:
                return Reply("workspace", "Pages are numbered from 1. Which page do you mean?")
            chunks = sorted((p for p in payloads if int(p["page_number"]) in pages and p["source_name"] in scope), key=_chunk_order)
            if not chunks:
                page = pages[0]
                if all(page > last_page[n] for n in scope):
                    if len(scope) == 1:
                        return Reply("workspace", f"**{_md(scope[0])}** has no indexed text after page {last_page[scope[0]]}.")
                    longest = max(scope, key=lambda n: last_page[n])
                    return Reply("workspace", f"None of the documents has indexed text on page {page}. "
                                              f"The furthest is **{_md(longest)}**, up to page {last_page[longest]}.")
                where = f" of **{_md(scope[0])}**" if len(scope) == 1 else ""
                return Reply("workspace", f"Page {page}{where} has no text I can read.")
            return _answer_reply(generate_answer(asked, chunks[:CONTEXT_LIMIT], services.answer_model))

        query = route.query or message  # retrieval uses the standalone reading
        hits = retriever.retrieve(query, document=route.document)
    finally:
        embed_cache.close()
    touch_session(services.client, collection)
    return _answer_reply(generate_answer(route.query or asked, [h.payload for h in hits], services.answer_model))


def redact(message: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        message = message.replace(secret, "***")
    return message
