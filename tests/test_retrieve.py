import zlib

import pytest
from qdrant_client import QdrantClient

from rag.bm25 import tokenize
from rag.chunk import Chunk
from rag.config import EMBED_DIM, EMBED_TASK_DOCUMENT
from rag.embed import EmbeddingCache, embed_texts
from rag.retrieve import Hit, Retriever, fuse, rrf
from rag.store import ensure_collection, replace_document

CITATION_FIELDS = {"chunk_id", "document_id", "source_name", "page_number", "section_path", "content_type", "text"}


# --- RRF -------------------------------------------------------------------


def test_rrf_single_ranking_uses_reciprocal_ranks():
    assert rrf([["a", "b"]], k=60) == [("a", 1 / 61), ("b", 1 / 62)]


def test_rrf_item_in_both_rankings_wins():
    fused = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [key for key, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_ties_keep_first_seen_order_and_k_is_applied():
    assert rrf([["a"], ["b"]], k=0) == [("a", 1.0), ("b", 1.0)]


def hit(chunk_id, **scores):
    return Hit({"chunk_id": chunk_id, "text": chunk_id, "page_number": 7, "section_path": ["S"]}, **scores)


def test_fuse_uses_ranks_not_raw_scores():
    dense = [hit("x", dense_score=0.99, dense_rank=1), hit("y", dense_score=0.98, dense_rank=2)]
    sparse = [hit("y", bm25_score=0.001, bm25_rank=1), hit("z", bm25_score=900.0, bm25_rank=2)]
    fused = fuse(dense, sparse)
    assert [h.chunk_id for h in fused] == ["y", "x", "z"]  # z's huge BM25 score is irrelevant

    rescaled = fuse(
        [hit("x", dense_score=0.1, dense_rank=1), hit("y", dense_score=0.0, dense_rank=2)],
        [hit("y", bm25_score=5e6, bm25_rank=1), hit("z", bm25_score=1e-9, bm25_rank=2)],
    )
    assert [(h.chunk_id, h.rrf_score) for h in rescaled] == [(h.chunk_id, h.rrf_score) for h in fused]


def test_fuse_merges_stage_scores_keeps_payload_and_limits_to_top_n():
    dense = [hit(f"d{i}", dense_score=1 - i / 100, dense_rank=i + 1) for i in range(20)]
    sparse = [hit("d3", bm25_score=4.2, bm25_rank=1)] + [hit(f"s{i}", bm25_score=1.0, bm25_rank=i + 2) for i in range(19)]
    fused = fuse(dense, sparse, top_n=10)
    assert len(fused) == 10
    top = fused[0]
    assert top.chunk_id == "d3"
    assert (top.dense_rank, top.dense_score, top.bm25_rank, top.bm25_score) == (4, 0.97, 1, 4.2)
    assert top.payload["page_number"] == 7


# --- Retriever (in-memory Qdrant, deterministic fake services) ----------------


def hash_embed(texts):
    """Bag-of-tokens hashed into EMBED_DIM buckets: similar wording -> similar vectors."""
    vectors = []
    for text in texts:
        v = [0.0] * EMBED_DIM
        for token in tokenize(text):
            v[zlib.crc32(token.encode()) % EMBED_DIM] += 1.0
        v[0] += 1e-3  # never a zero vector
        vectors.append(v)
    return vectors


class FakeReranker:
    """Scores documents by position in reverse, so final order differs from RRF order."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, documents, top_n):
        self.calls.append((query, list(documents), top_n))
        return [(i, float(i)) for i in range(len(documents))][::-1][:top_n]


def make_chunks(texts, doc_id="doc1"):
    return [
        Chunk(f"{doc_id}-p{i + 1}-{i}", doc_id, f"{doc_id}.pdf", i + 1, ("Handbook", f"Section {i}"), "text", t)
        for i, t in enumerate(texts)
    ]


def load(client, cache, chunks, doc_id="doc1"):
    vectors, _ = embed_texts([c.text for c in chunks], EMBED_TASK_DOCUMENT, hash_embed, cache)
    replace_document(client, doc_id, chunks, vectors)


@pytest.fixture
def env(tmp_path):
    client = QdrantClient(":memory:")
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    ensure_collection(client)
    yield client, cache
    cache.close()
    client.close()


def test_pipeline_stage_limits_metadata_and_rerank_order(env):
    client, cache = env
    load(client, cache, make_chunks([f"travel allowance rule number {i} for trip {i}" for i in range(30)]))
    reranker = FakeReranker()
    retriever = Retriever(client, hash_embed, cache, reranker)

    dense = retriever.dense_search("travel allowance")
    sparse = retriever.bm25_search("travel allowance")
    assert len(dense) == 20 and len(sparse) == 20

    direct = client.query_points("documents", query=hash_embed(["travel allowance"])[0], limit=20).points
    assert [h.dense_score for h in dense] == pytest.approx([p.score for p in direct])  # real Qdrant scores

    hits = retriever.retrieve("travel allowance")
    (query, documents, top_n), = reranker.calls
    assert query == "travel allowance" and len(documents) == 10 and top_n == 5
    assert len(hits) == 5
    assert [h.rerank_score for h in hits] == sorted((h.rerank_score for h in hits), reverse=True)
    assert [h.payload["text"] for h in hits] == documents[::-1][:5]  # reranker order, not RRF order
    for h in hits:
        assert CITATION_FIELDS <= h.payload.keys()
        assert h.rrf_score is not None and h.payload["section_path"][0] == "Handbook"


def test_hybrid_recovers_keyword_match_and_multilingual_bm25(env):
    client, cache = env
    load(
        client,
        cache,
        make_chunks(
            [
                "Employees receive 24 days of paid annual leave.",
                "सभी कर्मचारियों को हर 90 दिनों में पासवर्ड बदलना अनिवार्य है।",
                "Quarterly revenue table for FY2025.",
            ]
        ),
    )
    retriever = Retriever(client, hash_embed, cache, lambda q, docs, n: [(i, 1.0 - i / 10) for i in range(len(docs))][:n])

    (top_hi, *_) = retriever.retrieve("पासवर्ड कितने दिनों में बदलना है")
    assert top_hi.payload["page_number"] == 2 and top_hi.bm25_rank == 1

    (top_en, *_) = retriever.retrieve("paid annual leave")
    assert top_en.payload["page_number"] == 1 and top_en.dense_rank == 1


def test_empty_collection_returns_nothing_without_calling_reranker(env):
    client, cache = env
    reranker = FakeReranker()
    assert Retriever(client, hash_embed, cache, reranker).retrieve("anything") == []
    assert reranker.calls == []


def test_empty_query_rejected(env):
    client, cache = env
    with pytest.raises(ValueError):
        Retriever(client, hash_embed, cache, FakeReranker()).retrieve("   ")


def test_refresh_bm25_sees_newly_ingested_chunks(env):
    client, cache = env
    retriever = Retriever(client, hash_embed, cache, FakeReranker())
    assert retriever.bm25_search("reimbursement") == []
    load(client, cache, make_chunks(["expense reimbursement within 30 days"], doc_id="doc2"), doc_id="doc2")
    retriever.refresh_bm25()
    assert [h.payload["document_id"] for h in retriever.bm25_search("reimbursement")] == ["doc2"]


def test_query_embeddings_use_the_cache(env):
    client, cache = env
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return hash_embed(texts)

    retriever = Retriever(client, counting_embed, cache, FakeReranker())
    retriever.dense_search("same question")
    retriever.dense_search("same question")
    assert calls == [["same question"]]
