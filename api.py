"""Thin HTTP API over the RAG pipeline, for a UI. Prototype: one shared workspace, no authentication.

    uv run uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

FastAPI is only an adapter here: ingestion, retrieval, reranking, generation, prompts and
citation validation all live in rag/ and are called unchanged through rag.session.

Workspace: every client shares one Qdrant collection, API_COLLECTION (default "documents",
the collection the CLI also writes to). There is no per-client isolation and no access
control: do not expose this API publicly. Keep it on localhost behind a reverse proxy with
TLS and access control; the web frontend reaches it server-side.

Ingestion runs in the background on exactly one worker thread: PDFs take minutes, far past
proxy timeouts, and PyMuPDF must not run on several threads at once. At most one upload runs
and one waits (QUEUE_CAPACITY); further uploads get 429, so at most two PDFs are held in memory.
Job state (queued, processing, failed, empty and deleting documents) is kept in memory, so the
API must run as a single process (--workers 1). A restart forgets it; stored documents are then
re-derived from Qdrant, where a document counts as ready only if its write is verifiably
complete (rag.store.list_documents), so a write cut short by a failure or restart shows as
"incomplete" and can be uploaded again.
"""

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http.exceptions import ApiException

from rag import session
from rag.config import QDRANT_COLLECTION
from rag.extract import document_id_for
from rag.ingest import IngestReport
from rag.store import delete_document, list_documents

log = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_MB = 200
QUEUE_CAPACITY = 2  # one running ingestion plus one waiting
QUEUE_FULL = "The ingestion queue is full. Try again when the current upload has finished."
DocumentId = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]  # sha256(PDF bytes)[:16], as in rag.extract


# --- response models ------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    qdrant: bool
    collection: str


class IngestionStatus(BaseModel):
    stage: str  # queued, then extract | understand | chunk | embed | store | setup | finish | done
    pages: int
    chunks: int
    new_embeddings: int
    error: str | None  # exception class name only


class DocumentSummary(BaseModel):
    document_id: str
    source_name: str
    # ready: verifiably complete in Qdrant; incomplete: stored points from an unfinished write (upload again);
    # empty: processed, but no text could be extracted.
    status: Literal["ready", "incomplete", "queued", "processing", "failed", "empty"]
    chunks: int
    pages_with_chunks: int
    content_types: dict[str, int]
    ingestion: IngestionStatus | None = None  # set for queued, processing, failed and empty uploads


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


class IngestAccepted(BaseModel):
    document_id: str
    source_name: str
    status: Literal["queued", "ready"]


class DeleteResponse(BaseModel):
    document_id: str
    deleted_points: int


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    document: str  # as written in the answer: [document, Page N]
    document_id: str
    page: int


class Source(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    page_number: int
    section_path: list[str]
    content_type: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    sources: list[Source]  # the chunks behind the citations; Answer.raw_output is never exposed


# --- background ingestion -------------------------------------------------------------

BUSY = {"queued": "This document is already queued or being processed.",
        "processing": "This document is already queued or being processed.",
        "deleting": "This document is being deleted."}


@dataclass
class Job:
    source_name: str
    report: IngestReport | None  # live while queued/processing; None for a deletion in progress
    # queued -> processing -> (removed when ready) | failed | empty. "deleting" blocks uploads of the
    # document until its points are gone. A document with no chunks stores no points, so Qdrant
    # cannot list it: its job is kept as "empty".
    status: Literal["queued", "processing", "failed", "empty", "deleting"] = "queued"


def ingestion_status(job: Job) -> IngestionStatus:
    report = job.report
    return IngestionStatus(
        stage="queued" if job.status == "queued" else report.stage,  # a waiting job has not started extracting
        pages=report.pages,
        chunks=sum(report.chunks_by_type.values()),
        new_embeddings=report.embedding.new,
        error=report.error,
    )


def safe_filename(name: str | None) -> str:
    base = PurePosixPath((name or "").replace("\\", "/")).name.strip()
    return base or "upload.pdf"


def create_app(
    build_services: Callable[[], session.Services] = session.build_services,
    collection: str | None = None,
    max_upload_mb: int | None = None,
) -> FastAPI:
    state: dict = {}
    jobs: dict[str, Job] = {}
    lock = threading.Lock()  # guards `jobs` and snapshots of the reports in it

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["services"] = build_services()  # also loads .env, so the settings below can come from it
        state["collection"] = collection or os.environ.get("API_COLLECTION", QDRANT_COLLECTION)
        state["max_bytes"] = (max_upload_mb or int(os.environ.get("API_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))) * 1024 * 1024
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        app.state.ingest_executor = executor
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Multimodal Enterprise RAG API (single-workspace prototype)", lifespan=lifespan)

    def services() -> session.Services:
        return state["services"]

    def redacted(exc: Exception) -> str:
        return session.redact(f"{type(exc).__name__}: {exc}", services().secrets)

    def stored_documents():
        try:
            return list_documents(services().client, state["collection"])
        except Exception as exc:
            raise HTTPException(503, f"Vector database unavailable: {redacted(exc)}") from exc

    def run_ingestion(document_id: str, data: bytes, source_name: str, report: IngestReport) -> None:
        with lock:
            jobs[document_id].status = "processing"
        try:
            session.ingest_upload(services(), state["collection"], data, source_name, report=report)
        except Exception:  # the report already holds the failed stage and exception class
            log.warning("ingestion of %s failed at %s (%s)", document_id, report.stage, report.error)
            with lock:
                jobs[document_id].status = "failed"
            return
        with lock:
            if report.storage.upserted:
                jobs.pop(document_id, None)  # now listed from Qdrant
            else:
                jobs[document_id].status = "empty"

    @app.get("/api/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        try:  # Qdrant only: never Gemini, Jina or MiniMax
            services().client.get_collections()
            ok = True
        except Exception:
            ok = False
        if not ok:
            response.status_code = 503
        return HealthResponse(status="ok" if ok else "unavailable", qdrant=ok, collection=state["collection"])

    @app.get("/api/documents", response_model=DocumentList)
    def documents() -> DocumentList:
        stored = {d.document_id: d for d in stored_documents()}
        with lock:
            deleting = {doc_id for doc_id, job in jobs.items() if job.status == "deleting"}
            active = {doc_id: (job.source_name, job.status, ingestion_status(job))
                      for doc_id, job in jobs.items() if job.status != "deleting"}
        listed = [
            DocumentSummary(document_id=d.document_id, source_name=d.source_name,
                            status="ready" if d.complete else "incomplete", chunks=d.chunks,
                            pages_with_chunks=d.pages_with_chunks, content_types=d.content_types)
            for d in stored.values() if d.document_id not in active and d.document_id not in deleting
        ]
        for doc_id, (source_name, status, ingestion) in active.items():
            partial = stored.get(doc_id)  # a failed or running store stage can leave some points behind
            listed.append(DocumentSummary(
                document_id=doc_id, source_name=source_name, status=status,
                chunks=partial.chunks if partial else 0,
                pages_with_chunks=partial.pages_with_chunks if partial else 0,
                content_types=partial.content_types if partial else {},
                ingestion=ingestion,
            ))
        return DocumentList(documents=sorted(listed, key=lambda d: (d.source_name, d.document_id)))

    @app.post(
        "/api/documents",
        status_code=202,
        response_model=IngestAccepted,
        responses={
            200: {"model": IngestAccepted, "description": "Already stored and complete; nothing was re-processed"},
            429: {"description": QUEUE_FULL},
        },
    )
    def upload(file: UploadFile, response: Response) -> IngestAccepted:
        data = file.file.read(state["max_bytes"] + 1)
        if len(data) > state["max_bytes"]:
            raise HTTPException(413, f"File is larger than the {state['max_bytes'] // (1024 * 1024)} MB upload limit.")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(415, "Only PDF files are accepted.")
        document_id = document_id_for(data)

        with lock:
            job = jobs.get(document_id)
            if job and job.status in BUSY:
                raise HTTPException(409, BUSY[job.status])
            retry = job is not None  # failed (maybe partial points) or empty: ingest it again
        stored = stored_documents()
        existing = next((d for d in stored if d.document_id == document_id), None)
        if existing and existing.complete and not retry:
            with lock:  # a deletion may have started since the first check: never report it as ready
                job = jobs.get(document_id)
                if job and job.status in BUSY:
                    raise HTTPException(409, BUSY[job.status])
            response.status_code = 200
            return IngestAccepted(document_id=document_id, source_name=existing.source_name, status="ready")

        with lock:  # check and register together, so concurrent requests cannot queue the same document twice
            job = jobs.get(document_id)
            if job and job.status in BUSY:
                raise HTTPException(409, BUSY[job.status])
            if sum(j.status in ("queued", "processing") for j in jobs.values()) >= QUEUE_CAPACITY:
                raise HTTPException(429, QUEUE_FULL)
            taken = {d.source_name for d in stored if d.document_id != document_id}
            taken |= {j.source_name for doc_id, j in jobs.items() if doc_id != document_id}
            source_name = existing.source_name if existing else session.unique_source_name(safe_filename(file.filename), taken)
            report = IngestReport(source_name)
            jobs[document_id] = Job(source_name, report)
        app.state.ingest_executor.submit(run_ingestion, document_id, data, source_name, report)
        return IngestAccepted(document_id=document_id, source_name=source_name, status="queued")

    @app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
    def delete(document_id: DocumentId) -> DeleteResponse:
        with lock:
            job = jobs.get(document_id)
            if job and job.status in BUSY:
                raise HTTPException(409, BUSY[job.status])
            had_job = job is not None  # a failed or empty upload
            jobs[document_id] = Job(job.source_name if job else "", None, "deleting")  # blocks uploads meanwhile
        try:
            deleted = delete_document(services().client, document_id, state["collection"])
        except Exception as exc:
            raise HTTPException(503, f"Vector database unavailable: {redacted(exc)}") from exc
        finally:
            with lock:
                jobs.pop(document_id, None)
        if not deleted and not had_job:
            raise HTTPException(404, "No such document.")
        return DeleteResponse(document_id=document_id, deleted_points=deleted)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            answer, _ = session.answer_question(services(), state["collection"], request.question)
        except ApiException as exc:  # Qdrant: connection failures and error responses
            raise HTTPException(503, f"Vector database unavailable: {redacted(exc)}") from exc
        except Exception as exc:  # embedding, reranking or answer model
            raise HTTPException(502, f"Could not answer the question: {redacted(exc)}") from exc
        ids = {(s["source_name"], int(s["page_number"])): s["document_id"] for s in answer.sources}
        return ChatResponse(
            answer=answer.text,
            abstained=answer.abstained,
            citations=[Citation(document=doc, document_id=ids[(doc, page)], page=page) for doc, page in answer.citations],
            sources=[Source(**{field: s[field] for field in Source.model_fields}) for s in answer.sources],
        )

    return app


app = create_app()
