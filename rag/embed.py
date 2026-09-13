"""Gemini Embedding 2 with a persistent SQLite cache.

A text is embedded only if its (model, dim, task, text) key is not cached, so
re-ingesting a document never re-embeds it and an interrupted run resumes where
it stopped. Rate limits (429) and transient server errors are retried by the
SDK with exponential backoff and jitter.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Callable, Sequence
from pathlib import Path

from google import genai
from google.genai import types

from rag.config import EMBED_BATCH_SIZE, EMBED_DIM, GEMINI_EMBED_MODEL

EmbedBatch = Callable[[list[str]], list[list[float]]]

RETRY = types.HttpRetryOptions(
    attempts=8,
    initial_delay=2.0,
    max_delay=60.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)


def gemini_embedder(api_key: str, task_type: str) -> EmbedBatch:
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=RETRY))
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        # One Content per text: a single Content with many parts would be embedded as one vector.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=contents, config=config)
        return [e.values for e in result.embeddings]

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
) -> tuple[list[list[float]], int]:
    """Return (vectors aligned with texts, number of texts newly embedded)."""
    keys = [cache_key(t, task_type) for t in texts]
    vectors = cache.get_many(set(keys))
    pending = list({k: t for k, t in zip(keys, texts) if k not in vectors}.items())  # dedupes

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        result = embed_batch([text for _, text in batch])
        if len(result) != len(batch) or any(len(v) != EMBED_DIM for v in result):
            raise RuntimeError(
                f"Embedding API returned {len(result)} vectors for {len(batch)} texts (expected dim {EMBED_DIM})"
            )
        new = {key: vector for (key, _), vector in zip(batch, result)}
        cache.put_many(new)
        vectors.update(new)

    return [vectors[k] for k in keys], len(pending)
