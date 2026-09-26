"""Qdrant dense-vector store.

Point ids are UUIDv5 of the chunk id, so re-ingesting the same document
overwrites the same points. The payload carries all citation metadata.
"""

import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict

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
    points = [
        models.PointStruct(id=pid, vector=list(vector), payload=asdict(c) | {"section_path": list(c.section_path)})
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


def document_point_count(client: QdrantClient, document_id: str, name: str = QDRANT_COLLECTION) -> int:
    only_document = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])
    return client.count(name, count_filter=only_document, exact=True).count


def iter_payloads(client: QdrantClient, name: str = QDRANT_COLLECTION) -> Iterator[dict]:
    offset = None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            yield point.payload
        if offset is None:
            return
