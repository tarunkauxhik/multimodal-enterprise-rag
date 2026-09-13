import pytest
from qdrant_client import QdrantClient, models

from rag.bm25 import build_bm25, tokenize
from rag.chunk import Chunk
from rag.config import EMBED_DIM, QDRANT_COLLECTION
from rag.embed import EmbeddingCache
from rag.ingest import ingest_pdf
from rag.store import ensure_collection, iter_payloads, point_id, replace_document

CITATION_FIELDS = {"chunk_id", "document_id", "source_name", "page_number", "section_path", "content_type", "text"}


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        return [[1.0, float(len(t))] + [0.5] * (EMBED_DIM - 2) for t in texts]


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def chunk(doc_id, n, text="text"):
    return Chunk(f"{doc_id}-p1-{n}", doc_id, "f.pdf", 1, ("S",), "text", f"{text} {n}")


def test_ingest_is_idempotent_and_payload_has_citation_metadata(client, sample_pdf, tmp_path):
    cache, fake = EmbeddingCache(tmp_path / "e.sqlite"), FakeEmbedder()
    first = ingest_pdf(sample_pdf, "sample.pdf", client=client, embed_batch=fake, cache=cache)
    count = client.count(QDRANT_COLLECTION).count
    assert first.chunks == count > 0 and first.newly_embedded == fake.calls > 0

    second = ingest_pdf(sample_pdf, "sample.pdf", client=client, embed_batch=fake, cache=cache)
    assert second.document_id == first.document_id
    assert second.newly_embedded == 0 and fake.calls == first.newly_embedded
    assert client.count(QDRANT_COLLECTION).count == count

    points, _ = client.scroll(QDRANT_COLLECTION, limit=100, with_payload=True)
    for p in points:
        assert CITATION_FIELDS <= p.payload.keys()
        assert p.payload["document_id"] == first.document_id
        assert str(p.id) == point_id(p.payload["chunk_id"])
        assert isinstance(p.payload["section_path"], list)
    table = next(p.payload for p in points if p.payload["content_type"] == "table")
    assert table["page_number"] == 1 and table["section_path"][-1] == "2. Financials"


def test_reingest_removes_stale_chunks_only_for_that_document(client):
    ensure_collection(client)
    vec = [[1.0] * EMBED_DIM]
    replace_document(client, "docA", [chunk("docA", i) for i in range(3)], vec * 3)
    replace_document(client, "docB", [chunk("docB", 0)], vec)

    replace_document(client, "docA", [chunk("docA", 0)], vec)
    ids = sorted(p["chunk_id"] for p in iter_payloads(client))
    assert ids == ["docA-p1-0", "docB-p1-0"]

    replace_document(client, "docA", [], [])
    assert [p["chunk_id"] for p in iter_payloads(client)] == ["docB-p1-0"]


def test_ensure_collection_rejects_wrong_dimensions(client):
    client.create_collection(
        QDRANT_COLLECTION, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    with pytest.raises(RuntimeError, match="expected size=768"):
        ensure_collection(client)


def test_bm25_built_from_qdrant_payloads_ranks_matching_chunk(client, sample_pdf, tmp_path):
    ingest_pdf(sample_pdf, "sample.pdf", client=client, embed_batch=FakeEmbedder(), cache=EmbeddingCache(tmp_path / "e.sqlite"))
    index = build_bm25(iter_payloads(client))
    scores = index.bm25.get_scores(tokenize("outlook next year plans"))
    best = index.payloads[max(range(len(scores)), key=scores.__getitem__)]
    assert best["page_number"] == 2 and best["section_path"][-1] == "3. Outlook"


def test_bm25_empty_corpus():
    assert build_bm25([]).bm25 is None
