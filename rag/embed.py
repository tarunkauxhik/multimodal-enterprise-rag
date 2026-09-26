"""Gemini Embedding 2 with a persistent SQLite cache.

A text is embedded only if its (model, dim, task, text) key is not cached, so
re-ingesting a document never re-embeds it and an interrupted run resumes where
it stopped. Rate limits (429) and transient server errors are retried by the
SDK with exponential backoff and jitter; transport-level failures (dropped
connections, TLS errors) never reach that layer and are retried here, with the
same backoff the Jina and MiniMax clients use.

embed_texts can fill an EmbedStats with cache hits, new embeddings, batches and
our transport retries. The SDK's own 429/5xx retries are internal to google-genai
and are not visible here; they only show up as time.
"""

import hashlib
import random
import sqlite3
import time
from array import array
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import httpx
from google import genai
from google.genai import types

from rag.config import EMBED_BATCH_SIZE, EMBED_DIM, GEMINI_EMBED_MODEL

EmbedBatch = Callable[[list[str]], list[list[float]]]


@dataclass
class EmbedStats:
    texts: int = 0
    unique: int = 0
    cached: int = 0
    new: int = 0  # committed to the cache, so it survives a failure later in the run
    batches_attempted: int = 0
    batches_succeeded: int = 0
    transport_retries: int = 0  # our retry loop only; the SDK's own HTTP retries are not visible


# The embedder is shared across Streamlit session threads, so retries are counted into the stats of
# whichever embed_texts call is running in the current thread/context, never into a shared counter.
_active_stats: ContextVar[EmbedStats | None] = ContextVar("active_embed_stats", default=None)

RETRY = types.HttpRetryOptions(
    attempts=8,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)


def gemini_embedder(
    api_key: str,
    task_type: str,
    *,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> EmbedBatch:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=RETRY))
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        # One Content per text: a single Content with many parts would be embedded as one vector.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        error = ""
        for attempt in range(attempts):
            try:
                result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=contents, config=config)
            except httpx.TransportError as exc:
                # A dropped connection or TLS error carries no HTTP status, so RETRY above never sees it.
                error = type(exc).__name__
            else:
                return [e.values for e in result.embeddings]
            if attempt + 1 < attempts:
                if (stats := _active_stats.get()) is not None:
                    stats.transport_retries += 1
                sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError(f"Gemini embedding failed: {error}".replace(api_key, "***"))

    return embed_batch


def cache_key(text: str, task_type: str) -> str:
    return hashlib.sha256(f"{GEMINI_EMBED_MODEL}\n{EMBED_DIM}\n{task_type}\n{text}".encode()).hexdigest()


class EmbeddingCache:
    """Vectors stored as float32 blobs keyed by cache_key."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        found = {}
        keys = list(keys)
        for i in range(0, len(keys), 500):  # stay under SQLite's bound-parameter limit
            part = keys[i : i + 500]
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(part))})", part
            )
            found.update((key, array("f", blob).tolist()) for key, blob in rows)
        return found

    def put_many(self, vectors: dict[str, Sequence[float]]) -> None:
        with self._db:  # commit per batch so progress survives a crash
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                [(key, array("f", vector).tobytes()) for key, vector in vectors.items()],
            )

    def close(self) -> None:
        self._db.close()


def embed_texts(
    texts: Sequence[str],
    task_type: str,
    embed_batch: EmbedBatch,
    cache: EmbeddingCache,
    batch_size: int = EMBED_BATCH_SIZE,
    stats: EmbedStats | None = None,
) -> tuple[list[list[float]], int]:
    """Return (vectors aligned with texts, number of texts newly embedded).

    `stats`, if given, is updated as the work happens, so a caller still holds accurate
    partial progress when an exception escapes.
    """
    stats = stats if stats is not None else EmbedStats()
    keys = [cache_key(t, task_type) for t in texts]
    vectors = cache.get_many(set(keys))
    pending = list({k: t for k, t in zip(keys, texts) if k not in vectors}.items())  # dedupes
    stats.texts += len(texts)
    stats.unique += len(set(keys))
    stats.cached += len(vectors)

    token = _active_stats.set(stats)
    try:
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            stats.batches_attempted += 1
            result = embed_batch([text for _, text in batch])
            if len(result) != len(batch) or any(len(v) != EMBED_DIM for v in result):
                raise RuntimeError(
                    f"Embedding API returned {len(result)} vectors for {len(batch)} texts (expected dim {EMBED_DIM})"
                )
            new = {key: vector for (key, _), vector in zip(batch, result)}
            cache.put_many(new)
            vectors.update(new)
            stats.batches_succeeded += 1
            stats.new += len(batch)
    finally:
        _active_stats.reset(token)

    return [vectors[k] for k in keys], len(pending)
