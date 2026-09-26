"""Qdrant dense-vector store.

Point ids are UUIDv5 of the chunk id, so re-ingesting the same document
overwrites the same points. The payload carries all citation metadata.
"""

import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient, models

from rag.chunk import Chunk
from rag.config import EMBED_DIM, QDRANT_COLLECTION

UPSERT_BATCH = 256
# Fixed namespace: changing it changes every point id.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "multimodal-enterprise-rag/chunks")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, name: str = QDRANT_COLLECTION) -> None:
    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        if vectors.size != EMBED_DIM or vectors.distance != models.Distance.COSINE:
            raise RuntimeError(
                f"Collection {name!r} has size={vectors.size} distance={vectors.distance}; "
                f"expected size={EMBED_DIM} distance=Cosine"
            )
        return
    client.create_collection(
        name, vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
    )
    client.create_payload_index(name, "document_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "write_id", models.PayloadSchemaType.KEYWORD)  # retrieval filters on it


def replace_document(
    client: QdrantClient,
    document_id: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    name: str = QDRANT_COLLECTION,
    on_upserted: Callable[[int], None] | None = None,
) -> int | None:
    """Upsert a document's chunks, then delete any of its points not in this set.

    `on_upserted` receives the size of each upsert batch as soon as Qdrant acknowledges it.
    Returns the number of this document's stale points counted immediately before the delete,
    by a separate request (Qdrant's delete-by-filter reports no count): a concurrent writer to the
    same document could change the set in between. None if that count itself failed.
    """
    ids = [point_id(c.chunk_id) for c in chunks]
    # Completeness: every point of one write carries that write's id and chunk count, so an interrupted
    # write (too few points, or points from two writes) can be told apart from a finished one after a
    # restart (see list_documents). Retrieval and prompts never read these fields.
    completeness = {"chunk_total": len(chunks), "write_id": uuid.uuid4().hex}
    points = [
        models.PointStruct(
            id=pid, vector=list(vector), payload=asdict(c) | {"section_path": list(c.section_path)} | completeness
        )
        for pid, c, vector in zip(ids, chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(name, batch, wait=True)
        if on_upserted:
            on_upserted(len(batch))

    # ponytail: upsert-then-delete is not atomic; queries mid-ingest may briefly see stale chunks
    stale = models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))],
        must_not=[models.HasIdCondition(has_id=ids)] if ids else None,
    )
    try:
        stale_count = client.count(name, count_filter=stale, exact=True).count
    except Exception:  # the count is observability only: it must never stop the delete below
        stale_count = None
    client.delete(name, points_selector=models.FilterSelector(filter=stale), wait=True)
    return stale_count


def _only_document(document_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])


def document_point_count(client: QdrantClient, document_id: str, name: str = QDRANT_COLLECTION) -> int:
    return client.count(name, count_filter=_only_document(document_id), exact=True).count


def delete_document(client: QdrantClient, document_id: str, name: str = QDRANT_COLLECTION) -> int:
    """Delete every point of one document; return how many there were, counted just before the delete."""
    if not client.collection_exists(name):
        return 0
    count = document_point_count(client, document_id, name)
    if count:
        client.delete(name, points_selector=models.FilterSelector(filter=_only_document(document_id)), wait=True)
    return count


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    source_name: str
    chunks: int
    pages_with_chunks: int  # distinct pages holding at least one chunk; the PDF's page count is not stored
    content_types: dict[str, int]
    # True only when every point comes from one write and their number equals that write's chunk_total.
    # False for interrupted writes, leftover stale points, and points written before these fields existed.
    complete: bool


_LISTING_FIELDS = ["document_id", "source_name", "page_number", "content_type", "chunk_total", "write_id"]  # never the text


def complete_write_ids(payloads: Iterable[dict]) -> set[str]:
    """write_ids of the documents whose stored points are exactly one finished write.

    A document qualifies when all its points carry the same write_id and their number equals that
    write's chunk_total (both stamped by replace_document). An interrupted first write (too few
    points), an interrupted rewrite (two write_ids, even with the same chunk count), leftover stale
    points and points written before these fields existed never qualify.
    """
    totals: dict[str, set] = defaultdict(set)
    writes: dict[str, set] = defaultdict(set)
    counts: Counter = Counter()
    for p in payloads:
        doc = p["document_id"]
        totals[doc].add(p.get("chunk_total"))
        writes[doc].add(p.get("write_id"))
        counts[doc] += 1
    return {
        next(iter(writes[doc]))
        for doc in counts
        if len(totals[doc]) == 1 and len(writes[doc]) == 1 and None not in totals[doc] | writes[doc]
        and counts[doc] == next(iter(totals[doc]))
    }


def list_documents(client: QdrantClient, name: str = QDRANT_COLLECTION) -> list[StoredDocument]:
    """One entry per stored document, derived from point payloads (there is no separate registry)."""
    # ponytail: full scroll per call, fine for workspace-sized collections; use Qdrant facets if it gets slow
    if not client.collection_exists(name):
        return []
    payloads = list(iter_payloads(client, name, fields=_LISTING_FIELDS))
    complete = complete_write_ids(payloads)
    docs: dict[str, dict] = {}
    for p in payloads:
        doc = docs.setdefault(p["document_id"], {"source_name": p["source_name"], "pages": set(), "types": Counter(), "writes": set()})
        doc["pages"].add(p["page_number"])
        doc["types"][p["content_type"]] += 1
        doc["writes"].add(p.get("write_id"))
    return sorted(
        (
            StoredDocument(doc_id, d["source_name"], sum(d["types"].values()), len(d["pages"]),
                           dict(sorted(d["types"].items())), len(d["writes"]) == 1 and d["writes"] <= complete)
            for doc_id, d in docs.items()
        ),
        key=lambda d: (d.source_name, d.document_id),
    )


def iter_payloads(client: QdrantClient, name: str = QDRANT_COLLECTION, fields: list[str] | None = None) -> Iterator[dict]:
    """Every point's payload, or only `fields` of it."""
    offset = None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset, with_payload=fields or True, with_vectors=False)
        for point in points:
            yield point.payload
        if offset is None:
            return
