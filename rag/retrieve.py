"""Hybrid retrieval.

query -> Gemini query embedding (cached) -> Qdrant dense top 20
      +  BM25 top 20 (in-memory, built from Qdrant payloads)
      -> RRF over ranks (k=60) -> top 10 -> Jina rerank -> top 5

Every Hit carries the full chunk payload (document, page, section, text) and
the score and rank from each stage it appeared in.

Only complete documents are searched: both BM25 and dense search are limited to
points from finished writes (rag.store.complete_write_ids), so a document whose
write was cut short by a failure or a restart never takes part.

Usage: uv run python -m rag.retrieve "your question"
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from rag.bm25 import Bm25Index, build_bm25
from rag.config import (
    BM25_TOP_K,
    DENSE_TOP_K,
    EMBED_CACHE_PATH,
    EMBED_TASK_QUERY,
    QDRANT_COLLECTION,
    RERANK_TOP_K,
    RRF_K,
    RRF_TOP_K,
    load_settings,
)
from rag.embed import EmbedBatch, EmbeddingCache, embed_texts, gemini_embedder
from rag.rerank import Rerank, jina_reranker
from rag.store import complete_write_ids, ensure_collection, iter_payloads


@dataclass
class Hit:
    payload: dict  # chunk_id, document_id, source_name, page_number, section_path, content_type, text
    dense_score: float | None = None  # Qdrant cosine similarity
    dense_rank: int | None = None  # 1-based
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.payload["chunk_id"]


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank), rank from 1.

    Only positions matter, never the rankers' raw scores. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def fuse(dense: list[Hit], sparse: list[Hit], top_n: int = RRF_TOP_K) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for hit in dense + sparse:
        m = merged.setdefault(hit.chunk_id, Hit(hit.payload))
        if hit.dense_rank is not None:
            m.dense_score, m.dense_rank = hit.dense_score, hit.dense_rank
        if hit.bm25_rank is not None:
            m.bm25_score, m.bm25_rank = hit.bm25_score, hit.bm25_rank
    fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h in sparse]])[:top_n]
    for chunk_id, score in fused:
        merged[chunk_id].rrf_score = score
    return [merged[chunk_id] for chunk_id, _ in fused]


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        embed_query: EmbedBatch,
        cache: EmbeddingCache,
        rerank: Rerank,
        collection: str = QDRANT_COLLECTION,
    ):
        self.client, self.embed_query, self.cache, self.rerank = client, embed_query, cache, rerank
        self.collection = collection
        ensure_collection(client, collection)
        self.bm25: Bm25Index = build_bm25([])
        self.complete_writes: set[str] = set()
        self.refresh_bm25()

    def refresh_bm25(self) -> None:
        """Rebuild BM25 and the set of retrievable writes from the chunks in Qdrant. Call after ingestion.

        Only documents whose stored points are one finished write (rag.store.complete_write_ids) are
        searched, by BM25 and dense search alike, so a write cut short by a failure or a restart is
        never retrieved. Derived from Qdrant on every refresh, so it holds across restarts.
        """
        payloads = list(iter_payloads(self.client, self.collection))
        self.complete_writes = complete_write_ids(payloads)
        self.bm25 = build_bm25(p for p in payloads if p.get("write_id") in self.complete_writes)

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[Hit]:
        (vector,), _ = embed_texts([query], EMBED_TASK_QUERY, self.embed_query, self.cache)
        if not self.complete_writes:
            return []
        # Filtered in Qdrant, so the top_k come from complete documents only. A write that starts after
        # the refresh has a new write_id, which is not in the set: its partial points stay invisible.
        complete_only = models.Filter(
            must=[models.FieldCondition(key="write_id", match=models.MatchAny(any=sorted(self.complete_writes)))]
        )
        points = self.client.query_points(
            self.collection, query=vector, query_filter=complete_only, limit=top_k, with_payload=True
        ).points
        return [Hit(p.payload, dense_score=p.score, dense_rank=rank) for rank, p in enumerate(points, start=1)]

    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[Hit]:
        return [
            Hit(payload, bm25_score=score, bm25_rank=rank)
            for rank, (payload, score) in enumerate(self.bm25.search(query, top_k), start=1)
        ]

    def rerank_hits(self, query: str, hits: list[Hit], top_n: int = RERANK_TOP_K) -> list[Hit]:
        if not hits:
            return []
        ranked = self.rerank(query, [h.payload["text"] for h in hits], min(top_n, len(hits)))
        out = []
        for index, score in sorted(ranked, key=lambda r: r[1], reverse=True)[:top_n]:
            hits[index].rerank_score = score
            out.append(hits[index])
        return out

    def retrieve(self, query: str) -> list[Hit]:
        query = query.strip()
        if not query:
            raise ValueError("Query is empty")
        return self.rerank_hits(query, fuse(self.dense_search(query), self.bm25_search(query)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval for a query.")
    parser.add_argument("query")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi output on Windows consoles/pipes

    settings = load_settings()
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    try:
        retriever = Retriever(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
            gemini_embedder(settings.gemini_api_key, EMBED_TASK_QUERY),
            cache,
            jina_reranker(settings.jina_api_key),
        )
        for i, hit in enumerate(retriever.retrieve(args.query), start=1):
            p = hit.payload
            print(
                f"{i}. [Page {p['page_number']}] {p['source_name']} > {' > '.join(p['section_path'])} | "
                f"rerank={hit.rerank_score:.3f} rrf={hit.rrf_score:.4f} "
                f"dense_rank={hit.dense_rank} bm25_rank={hit.bm25_rank}"
            )
            print("   " + p["text"][:160].replace("\n", " "))
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
