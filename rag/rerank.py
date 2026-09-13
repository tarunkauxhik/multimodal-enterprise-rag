"""Jina reranker client (jina-reranker-v3)."""

import random
import time
from collections.abc import Callable

import httpx

from rag.config import JINA_RERANK_MODEL

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# (query, documents, top_n) -> [(document index, relevance score)], best first
Rerank = Callable[[str, list[str], int], list[tuple[int, float]]]


def jina_reranker(
    api_key: str,
    *,
    attempts: int = 5,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Rerank:
    http = httpx.Client(timeout=timeout, transport=transport, headers={"Authorization": f"Bearer {api_key}"})

    def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        body = {
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        error = ""
        for attempt in range(attempts):
            try:
                resp = http.post(JINA_RERANK_URL, json=body)
            except httpx.TransportError as exc:
                error = type(exc).__name__
            else:
                if resp.status_code == 200:
                    results = sorted(resp.json()["results"], key=lambda r: r["relevance_score"], reverse=True)
                    return [(r["index"], float(r["relevance_score"])) for r in results]
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if attempt + 1 < attempts:
                sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError(f"Jina rerank failed: {error}".replace(api_key, "***"))

    return rerank
