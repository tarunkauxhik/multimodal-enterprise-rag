"""Ingest PDFs: extract -> selective M3 understanding -> chunk -> embed (cached) -> Qdrant.

Usage: uv run python -m rag.ingest path/to/file.pdf [more.pdf ...]

Idempotent: document and chunk ids are content-derived, embeddings and M3 page
results are cached, and re-ingesting a document replaces its points in place.
The BM25 index is rebuilt from Qdrant by the app, so ingestion does not touch it.

The CLI always writes to the shared QDRANT_COLLECTION ("documents"). The Streamlit
app calls ingest_pdf with its own per-session collection instead (rag.session),
so documents ingested here are not visible in the app.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from rag.chunk import chunk_document
from rag.config import (
    EMBED_CACHE_PATH,
    EMBED_TASK_DOCUMENT,
    QDRANT_COLLECTION,
    UNDERSTAND_CACHE_PATH,
    UNDERSTAND_MAX_TOKENS,
    load_settings,
)
from rag.embed import EmbedBatch, EmbeddingCache, embed_texts, gemini_embedder
from rag.extract import extract_pdf
from rag.generate import Complete, minimax_client
from rag.store import ensure_collection, replace_document
from rag.understand import UnderstandingCache, understand_document


@dataclass
class IngestResult:
    document_id: str
    source_name: str
    pages: int
    chunks: int
    newly_embedded: int
    understood_pages: list[int] = field(default_factory=list)  # enriched by M3
    failed_pages: list[int] = field(default_factory=list)  # M3 failed; fast extraction kept


def ingest_pdf(
    data: bytes,
    source_name: str,
    *,
    client: QdrantClient,
    embed_batch: EmbedBatch,
    cache: EmbeddingCache,
    complete: Complete | None = None,
    understanding_cache: UnderstandingCache | None = None,
    collection: str = QDRANT_COLLECTION,
    on_stage: Callable[[str], None] | None = None,
) -> IngestResult:
    """Ingest one PDF. Without `complete`, only the fast extraction path is used.

    `on_stage` receives a short human-readable message as each stage starts.
    """
    stage = on_stage or (lambda message: None)

    stage("Extracting text and layout")
    doc = extract_pdf(data, source_name)
    understood, failed = [], []
    if complete is not None:
        stage("Checking for figures, tables and scanned pages (MiniMax M3 where needed)")
        doc, report = understand_document(doc, data, complete, understanding_cache)
        understood, failed = report.understood, report.failed
    stage("Chunking by structure")
    chunks = chunk_document(doc)
    stage(f"Embedding {len(chunks)} chunks")
    vectors, newly_embedded = embed_texts([c.text for c in chunks], EMBED_TASK_DOCUMENT, embed_batch, cache)
    stage("Storing in the vector database")
    ensure_collection(client, collection)
    replace_document(client, doc.document_id, chunks, vectors, collection)
    return IngestResult(doc.document_id, source_name, len(doc.pages), len(chunks), newly_embedded, understood, failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embed_batch = gemini_embedder(settings.gemini_api_key, EMBED_TASK_DOCUMENT)
    complete = minimax_client(settings.minimax_api_key, settings.minimax_base_url, max_tokens=UNDERSTAND_MAX_TOKENS)
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    understanding_cache = UnderstandingCache(UNDERSTAND_CACHE_PATH)

    failed = 0
    try:
        for path in args.pdfs:
            try:
                r = ingest_pdf(
                    path.read_bytes(),
                    path.name,
                    client=client,
                    embed_batch=embed_batch,
                    cache=cache,
                    complete=complete,
                    understanding_cache=understanding_cache,
                )
            except (OSError, ValueError) as exc:  # bad file: report and continue with the rest
                print(f"FAILED {path}: {exc}", file=sys.stderr)
                failed += 1
                continue
            print(
                f"{r.source_name}: document_id={r.document_id} pages={r.pages} chunks={r.chunks} "
                f"newly_embedded={r.newly_embedded} m3_pages={r.understood_pages} m3_failed={r.failed_pages}"
            )
            if r.chunks == 0:
                print(f"  warning: no text extracted from {r.source_name}", file=sys.stderr)
    finally:
        cache.close()
        understanding_cache.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
