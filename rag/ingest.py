"""Ingest PDFs: extract -> selective M3 understanding -> chunk -> embed (cached) -> Qdrant.

Usage: uv run python -m rag.ingest path/to/file.pdf [more.pdf ...] [--json]

Idempotent: document and chunk ids are content-derived, embeddings and M3 page
results are cached, and re-ingesting a document replaces its points in place.
The BM25 index is rebuilt from Qdrant by the app, so ingestion does not touch it.

The CLI always writes to the shared QDRANT_COLLECTION ("documents"), which is also
the HTTP API's default workspace (api.py, through rag.session).

Observability: every ingestion fills an IngestReport (routing, chunks, embedding,
storage, per-stage timings). The caller may pass its own report so that it keeps the
partial progress when ingestion raises; the original exception is always re-raised
unchanged. The report holds counts, page numbers and exception class names only:
never document text, API keys or API error messages.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field
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
from rag.embed import EmbedBatch, EmbeddingCache, EmbedStats, embed_texts, gemini_embedder
from rag.extract import extract_pdf
from rag.generate import Complete, minimax_client
from rag.store import document_point_count, ensure_collection, replace_document
from rag.understand import RoutingStats, UnderstandingCache, plan_page, routing_stats, understand_document

STAGES = ("extract", "understand", "chunk", "embed", "store")
REASON_ORDER = ("scanned", "garbled", "legacy_font", "legacy_text", "figure", "table")
REASON_LABELS = {"table": "sparse table"}


@dataclass
class StoreStats:
    upserted: int = 0  # points in upsert batches Qdrant acknowledged, counted batch by batch
    stale_deleted: int | None = None  # stale points counted just before the delete; None if not counted
    # Snapshots counted right after this document's write (other writers may change them later);
    # None when the write did not complete or the count itself failed.
    document_points: int | None = None
    collection_points: int | None = None


@dataclass
class IngestReport:
    source_name: str
    document_id: str = ""
    pages: int = 0
    # The stage running, or the one that failed; "done" on success. Besides STAGES, rag.session uses
    # "setup" (opening caches) and "finish" (recording session activity after the write).
    stage: str = "extract"
    error: str | None = None  # exception class name only, never its message
    empty_pages: list[int] = field(default_factory=list)  # no extracted text at all
    routing: RoutingStats = field(default_factory=RoutingStats)
    chunks_by_type: dict[str, int] = field(default_factory=dict)
    embedding: EmbedStats = field(default_factory=EmbedStats)
    storage: StoreStats = field(default_factory=StoreStats)
    seconds: dict[str, float] = field(default_factory=dict)  # per stage, plus "total"

    def to_dict(self) -> dict:
        return asdict(self)

    def reset(self, source_name: str) -> None:
        """Back to a fresh report, keeping this object so a caller's reference stays valid."""
        self.__dict__.update(vars(IngestReport(source_name)))


@dataclass
class IngestResult:
    document_id: str
    source_name: str
    pages: int
    chunks: int
    newly_embedded: int
    understood_pages: list[int] = field(default_factory=list)  # enriched by M3
    failed_pages: list[int] = field(default_factory=list)  # M3 failed; fast extraction kept
    report: IngestReport | None = None


@contextmanager
def report_stage(report: IngestReport, stage: str) -> Iterator[None]:
    """Mark `stage` as running; on an exception record its class name and re-raise it unchanged."""
    report.stage = stage
    try:
        yield
    except Exception as exc:
        report.error = type(exc).__name__
        raise


def _snapshot(count: Callable[[], int]) -> int | None:
    """Post-write counts are observability only: a failed read must not fail a completed write."""
    try:
        return count()
    except Exception:
        return None


@contextmanager
def _timed(report: IngestReport, stage: str) -> Iterator[None]:
    report.stage = stage
    start = time.perf_counter()
    try:
        yield
    finally:  # timed even when the stage fails
        report.seconds[stage] = round(time.perf_counter() - start, 3)


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
    report: IngestReport | None = None,
) -> IngestResult:
    """Ingest one PDF. Without `complete`, only the fast extraction path is used.

    `on_stage` receives a short human-readable message as each stage starts. `report`, if
    given, is reset, then filled in as ingestion proceeds and keeps the partial progress on failure.
    """
    stage = on_stage or (lambda message: None)
    if report is None:
        report = IngestReport(source_name)
    else:
        report.reset(source_name)  # a reused report must not carry an old error or add to old counts
    started = time.perf_counter()
    understood, failed = [], []
    try:
        with _timed(report, "extract"):
            stage("Extracting text and layout")
            doc = extract_pdf(data, source_name)
            report.document_id, report.pages = doc.document_id, len(doc.pages)
            report.empty_pages = [p.number for p in doc.pages if not any(b.text.strip() for b in p.blocks)]

        with _timed(report, "understand"):
            # Planned on the fast extraction, so routing is reported even with M3 disabled.
            report.routing = routing_stats([plan_page(p) for p in doc.pages], m3_enabled=complete is not None)
            if complete is not None:
                stage("Checking for figures, tables and scanned pages (MiniMax M3 where needed)")
                doc, understanding = understand_document(doc, data, complete, understanding_cache)
                understood, failed = understanding.understood, understanding.failed
                report.routing.understood = len(understood)
                report.routing.failed = list(failed)
                report.routing.from_cache = understanding.from_cache

        with _timed(report, "chunk"):
            stage("Chunking by structure")
            chunks = chunk_document(doc)
            report.chunks_by_type = dict(sorted(Counter(c.content_type for c in chunks).items()))

        with _timed(report, "embed"):
            stage(f"Embedding {len(chunks)} chunks")
            vectors, newly_embedded = embed_texts(
                [c.text for c in chunks], EMBED_TASK_DOCUMENT, embed_batch, cache, stats=report.embedding
            )

        with _timed(report, "store"):
            stage("Storing in the vector database")
            ensure_collection(client, collection)

            def upserted(points: int) -> None:  # as each batch lands, so a partial write is reported as such
                report.storage.upserted += points

            report.storage.stale_deleted = replace_document(
                client, doc.document_id, chunks, vectors, collection, on_upserted=upserted
            )
            report.storage.document_points = _snapshot(lambda: document_point_count(client, doc.document_id, collection))
            report.storage.collection_points = _snapshot(lambda: client.count(collection, exact=True).count)
    except Exception as exc:
        report.error = type(exc).__name__
        raise  # the original exception, unchanged
    finally:
        report.seconds["total"] = round(time.perf_counter() - started, 3)
    report.stage = "done"
    return IngestResult(
        doc.document_id, source_name, len(doc.pages), len(chunks), newly_embedded, understood, failed, report
    )


def format_report(report: IngestReport) -> str:
    """Concise human-readable summary; only stages that started are shown."""
    head = f"{report.source_name}  id={report.document_id or '-'}  {report.pages} pages"
    head += f"  total {report.seconds.get('total', 0.0):.1f} s"
    if report.error:
        head += f"  FAILED at {report.stage} ({report.error})"
    lines = [head]
    r, e, s = report.routing, report.embedding, report.storage
    for name in STAGES:
        if name not in report.seconds:
            continue
        if name == "extract":
            detail = f"{report.pages} pages, {len(report.empty_pages)} empty"
        elif name == "understand":
            reasons = ", ".join(
                f"{REASON_LABELS.get(k, k)} {r.reasons[k]}" for k in REASON_ORDER if r.reasons.get(k)
            )
            detail = f"{r.routed_pages} to M3 ({reasons or 'no reasons'}; {sum(r.overlaps.values())} overlapping), "
            detail += f"{r.normal_pages} normal"
            if r.m3_enabled:
                detail += f"; understood {r.understood} ({r.from_cache} from cache), failed {len(r.failed)}"
                if r.failed:
                    detail += f" (pages {', '.join(map(str, r.failed))})"
            else:
                detail += "; M3 disabled"
        elif name == "chunk":
            by_type = ", ".join(f"{k} {v}" for k, v in report.chunks_by_type.items())
            detail = f"{sum(report.chunks_by_type.values())} ({by_type})" if by_type else "0"
        elif name == "embed":
            detail = (f"{e.texts} texts, {e.unique} unique: {e.new} new, {e.cached} cached; "
                      f"batches {e.batches_succeeded}/{e.batches_attempted}; transport retries {e.transport_retries}")
        elif report.error and report.stage == "store":
            detail = f"{s.upserted} upserted before the failure"
        else:
            def known(value: int | None) -> str:
                return "?" if value is None else str(value)

            detail = (f"{s.upserted} upserted, {known(s.stale_deleted)} stale deleted; snapshot after write: "
                      f"document {known(s.document_points)} points, collection {known(s.collection_points)}")
        lines.append(f"  {name:<11}{detail}  [{report.seconds[name]:.1f} s]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="print the ingestion reports as JSON instead of a summary")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi file names on Windows consoles

    # With --json, stdout carries the JSON and nothing else: everything written while ingesting goes to stderr.
    with _stdout_to_stderr() if args.json else nullcontext():
        reports, failed = _ingest_paths(args.pdfs, summary=not args.json)
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], ensure_ascii=False, indent=2))
    return 1 if failed else 0


@contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Route file descriptor 1 and sys.stdout to stderr: ours, PyMuPDF's own handle and native writes alike."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()  # buffered writes made through the original handle still go to stderr
        os.dup2(saved, 1)
        os.close(saved)


def _ingest_paths(pdfs: list[Path], summary: bool) -> tuple[list[IngestReport], int]:
    settings = load_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embed_batch = gemini_embedder(settings.gemini_api_key, EMBED_TASK_DOCUMENT)
    complete = minimax_client(settings.minimax_api_key, settings.minimax_base_url, max_tokens=UNDERSTAND_MAX_TOKENS)
    cache = EmbeddingCache(EMBED_CACHE_PATH)
    understanding_cache = UnderstandingCache(UNDERSTAND_CACHE_PATH)

    reports: list[IngestReport] = []
    failed = 0
    try:
        for path in pdfs:
            report = IngestReport(path.name)
            reports.append(report)
            try:
                r = ingest_pdf(
                    path.read_bytes(),
                    path.name,
                    client=client,
                    embed_batch=embed_batch,
                    cache=cache,
                    complete=complete,
                    understanding_cache=understanding_cache,
                    report=report,
                )
            except (OSError, ValueError) as exc:  # bad file: report and continue with the rest
                report.error = report.error or type(exc).__name__  # a read error happens before ingest_pdf runs
                print(f"FAILED {path}: {exc}", file=sys.stderr)
                failed += 1
                continue
            except Exception:
                print(format_report(report), file=sys.stderr)  # partial progress, then the original error
                raise
            if summary:
                print(format_report(report))
            if r.chunks == 0:
                print(f"  warning: no text extracted from {r.source_name}", file=sys.stderr)
    finally:
        cache.close()
        understanding_cache.close()
    return reports, failed


if __name__ == "__main__":
    sys.exit(main())
