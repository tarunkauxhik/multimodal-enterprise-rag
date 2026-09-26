"""HTTP API adapter (offline: in-memory Qdrant and fake services from conftest; real background worker)."""

import threading

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

import api as api_module
from rag import store as store_module
from rag.chunk import Chunk
from rag.config import EMBED_DIM
from rag.generate import ABSTAIN_MESSAGE
from rag.store import delete_document, ensure_collection, list_documents, point_id, replace_document
from tests.conftest import RecordingModel, build_text_pdf, hash_embed

WORKSPACE = "documents"
REPLY = "Revenue was 120 in 2025 [sample.pdf, Page 1]."


class CountingEmbedder:
    def __init__(self):
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        return hash_embed(texts)


class GatedEmbedder(CountingEmbedder):
    """Holds ingestion inside the embed stage until released, so tests can see a job while it runs."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, texts):
        self.entered.set()
        assert self.release.wait(20), "test never released the embedder"
        return super().__call__(texts)


@pytest.fixture
def api(fake_services):
    fake_services.embed_document = CountingEmbedder()
    app = api_module.create_app(build_services=lambda: fake_services, collection=WORKSPACE, max_upload_mb=1)
    with TestClient(app) as client:
        client.services = fake_services
        client.drain = lambda: app.state.ingest_executor.submit(lambda: None).result(timeout=60)  # one worker: FIFO
        yield client


def upload(api, data, name="sample.pdf"):
    return api.post("/api/documents", files={"file": (name, data, "application/pdf")})


def ingested(api, data, name="sample.pdf"):
    response = upload(api, data, name)
    assert response.status_code == 202, response.text
    api.drain()
    return response.json()


def listed(api):
    response = api.get("/api/documents")
    assert response.status_code == 200
    return response.json()["documents"]


# --- health -------------------------------------------------------------------


def test_health_checks_qdrant_only(api):
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": True, "collection": WORKSPACE}
    assert api.services.answer_model.calls == [] and api.services.understand_model.calls == []
    assert api.services.embed_document.texts == 0  # no Gemini, MiniMax or Jina call


def test_health_reports_qdrant_outage_as_503(api, monkeypatch):
    def down():
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(api.services.client, "get_collections", down)
    response = api.get("/api/health")
    assert response.status_code == 503 and response.json()["status"] == "unavailable"


# --- upload and background ingestion --------------------------------------------------------


def test_upload_is_accepted_then_completes_in_the_background(api, sample_pdf):
    response = upload(api, sample_pdf)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["source_name"] == "sample.pdf" and len(body["document_id"]) == 16

    api.drain()
    (doc,) = listed(api)
    assert doc["document_id"] == body["document_id"] and doc["status"] == "ready"
    assert doc["chunks"] > 0 and doc["pages_with_chunks"] == 2
    assert set(doc["content_types"]) == {"text", "table", "figure"} and doc["ingestion"] is None


def test_processing_job_shows_live_status_and_duplicates_get_409(api, sample_pdf):
    gated = GatedEmbedder()
    api.services.embed_document = gated
    accepted = upload(api, sample_pdf).json()
    assert gated.entered.wait(20)

    (doc,) = listed(api)
    assert doc["status"] == "processing"
    assert doc["ingestion"] == {"stage": "embed", "pages": 2, "chunks": 4, "new_embeddings": 0, "error": None}
    assert upload(api, sample_pdf).status_code == 409  # same document, still processing
    assert api.delete(f"/api/documents/{accepted['document_id']}").status_code == 409

    gated.release.set()
    api.drain()
    assert [d["status"] for d in listed(api)] == ["ready"]


def test_already_ready_duplicate_returns_200_without_reingesting(api, sample_pdf):
    first = ingested(api, sample_pdf)
    texts_embedded = api.services.embed_document.texts

    again = upload(api, sample_pdf, name="renamed-copy.pdf")
    assert again.status_code == 200
    assert again.json() == {"document_id": first["document_id"], "source_name": "sample.pdf", "status": "ready"}
    api.drain()
    assert api.services.embed_document.texts == texts_embedded and len(listed(api)) == 1


def test_non_pdf_upload_is_rejected_with_415(api):
    response = upload(api, b"not a pdf", name="notes.txt")
    assert response.status_code == 415 and listed(api) == []


def test_corrupt_pdf_fails_in_the_background_and_can_be_retried_or_removed(api):
    broken = b"%PDF-1.7\nthis is not really a pdf"
    first = upload(api, broken, name="broken.pdf")
    assert first.status_code == 202
    api.drain()
    (doc,) = listed(api)
    assert doc["status"] == "failed" and doc["ingestion"]["stage"] == "extract" and doc["ingestion"]["error"] == "ValueError"

    assert upload(api, broken, name="broken.pdf").status_code == 202  # a failed upload may be retried
    api.drain()
    removed = api.delete(f"/api/documents/{first.json()['document_id']}")
    assert removed.status_code == 200 and removed.json()["deleted_points"] == 0 and listed(api) == []


def test_upload_with_no_extractable_text_stays_listed_as_empty(api):
    """No chunks means no Qdrant points; without the job it would silently vanish from the listing."""
    blank = build_text_pdf(["A lone header line."])  # layout analysis drops it: nothing to chunk
    doc_id = ingested(api, blank, "blank.pdf")["document_id"]
    (doc,) = listed(api)
    assert (doc["document_id"], doc["status"], doc["chunks"]) == (doc_id, "empty", 0)
    assert doc["ingestion"]["stage"] == "done" and doc["ingestion"]["error"] is None
    assert upload(api, blank, "blank.pdf").status_code == 202  # not "ready": it may be retried
    api.drain()
    assert api.delete(f"/api/documents/{doc_id}").status_code == 200 and listed(api) == []


def test_upload_over_the_size_limit_is_rejected_with_413(api):
    too_big = b"%PDF-" + b"0" * (1024 * 1024)  # max_upload_mb=1 in the fixture
    response = upload(api, too_big, name="big.pdf")
    assert response.status_code == 413 and "1 MB" in response.json()["detail"] and listed(api) == []


# --- listing and deletion ---------------------------------------------------------------


def test_listing_keeps_names_unique_and_never_includes_chunk_text(api, sample_pdf):
    ingested(api, sample_pdf, "sample.pdf")
    ingested(api, build_text_pdf(["Line one of another report.", "Second line with more words.", "Third line closes it."]), "sample.pdf")
    documents = listed(api)
    assert [d["source_name"] for d in documents] == ["sample (2).pdf", "sample.pdf"]
    assert all(d["status"] == "ready" for d in documents)
    assert "text" not in {key for d in documents for key in d}
    assert "paragraph line" not in api.get("/api/documents").text


def test_list_documents_reads_only_metadata_fields(monkeypatch):
    client = QdrantClient(":memory:")
    ensure_collection(client, WORKSPACE)
    chunks = [Chunk(f"docA-p{p}-{i}", "docA", "a.pdf", p, ("S",), kind, "secret body text")
              for i, (p, kind) in enumerate([(1, "text"), (1, "table"), (3, "text")])]
    replace_document(client, "docA", chunks, [[1.0] * EMBED_DIM] * 3, WORKSPACE)
    requested = []
    real_scroll = client.scroll

    def recording_scroll(*args, **kwargs):
        requested.append(kwargs.get("with_payload"))
        return real_scroll(*args, **kwargs)

    monkeypatch.setattr(client, "scroll", recording_scroll)
    (doc,) = list_documents(client, WORKSPACE)
    assert (doc.document_id, doc.source_name, doc.chunks, doc.pages_with_chunks) == ("docA", "a.pdf", 3, 2)
    assert doc.content_types == {"table": 1, "text": 2}
    assert all("text" not in fields for fields in requested)  # chunk text is never loaded
    assert list_documents(client, "missing") == [] and delete_document(client, "docA", "missing") == 0
    client.close()


def test_delete_removes_every_point_then_404s(api, sample_pdf):
    doc_id = ingested(api, sample_pdf)["document_id"]
    chunks = listed(api)[0]["chunks"]
    response = api.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 200 and response.json() == {"document_id": doc_id, "deleted_points": chunks}
    assert listed(api) == []
    assert api.delete(f"/api/documents/{doc_id}").status_code == 404
    assert api.delete("/api/documents/not-a-document-id").status_code == 422


# --- chat ---------------------------------------------------------------------


def test_chat_returns_answer_with_validated_citations_and_sources(api, sample_pdf):
    doc_id = ingested(api, sample_pdf)["document_id"]
    response = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "abstained", "citations", "sources"}  # no ranks, scores or raw output
    assert body["answer"] == REPLY and body["abstained"] is False
    assert body["citations"] == [{"document": "sample.pdf", "document_id": doc_id, "page": 1}]
    assert body["sources"] and all(s["source_name"] == "sample.pdf" and s["page_number"] == 1 for s in body["sources"])
    assert set(body["sources"][0]) == {"chunk_id", "document_id", "source_name", "page_number", "section_path", "content_type", "text"}


def test_chat_abstains_on_an_empty_workspace_without_calling_the_model(api):
    response = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    assert response.status_code == 200
    assert response.json() == {"answer": ABSTAIN_MESSAGE, "abstained": True, "citations": [], "sources": []}
    assert api.services.answer_model.calls == []


def test_chat_abstains_when_the_answer_has_no_valid_citation(api, sample_pdf):
    ingested(api, sample_pdf)
    api.services.answer_model = RecordingModel("Revenue was 120 [other.pdf, Page 9].")
    body = api.post("/api/chat", json={"question": "What was revenue in 2025?"}).json()
    assert body["abstained"] is True and body["citations"] == [] and body["answer"] == ABSTAIN_MESSAGE


def test_chat_provider_errors_are_502_and_redacted(api, sample_pdf):
    ingested(api, sample_pdf)
    api.services.answer_model = RecordingModel(RuntimeError("MiniMax completion failed: HTTP 401 invalid key sk-test-secret"))
    response = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    assert response.status_code == 502
    assert "HTTP 401" in response.json()["detail"] and "sk-test-secret" not in response.text


def test_raw_model_output_never_reaches_the_response(api, sample_pdf):
    ingested(api, sample_pdf)
    api.services.answer_model = RecordingModel(f"<think>PRIVATE-REASONING</think>{REPLY}")
    grounded = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    api.services.answer_model = RecordingModel("UNGROUNDED-CLAIM with no citation")
    abstained = api.post("/api/chat", json={"question": "What was revenue in 2025?"})

    assert grounded.json()["answer"] == REPLY and abstained.json()["abstained"] is True
    for response in (grounded, abstained):
        assert "raw_output" not in response.json()
        assert "PRIVATE-REASONING" not in response.text and "UNGROUNDED-CLAIM" not in response.text


@pytest.mark.parametrize("payload", [{"question": "   "}, {"question": ""}, {}, {"question": "ok", "debug": True}])
def test_invalid_chat_requests_are_422(api, payload):
    assert api.post("/api/chat", json=payload).status_code == 422


# --- completeness, lifecycle races, queue bounds, error semantics ---------------------------

OTHER = ["Line one of another report.", "Second line with more words.", "Third line closes it."]
THIRD = ["A third report starts here.", "It has its own second line.", "And a closing third line."]
VECTOR = [1.0] * EMBED_DIM


def drain(app):
    app.state.ingest_executor.submit(lambda: None).result(timeout=60)


def test_partial_store_failure_is_never_ready_even_after_job_state_is_lost(fake_services, sample_pdf, monkeypatch):
    fake_services.embed_document = CountingEmbedder()
    monkeypatch.setattr(store_module, "UPSERT_BATCH", 2)  # the sample's 4 chunks -> 2 upsert batches
    real_upsert, calls = fake_services.client.upsert, []

    def second_batch_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("qdrant write failed")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(fake_services.client, "upsert", second_batch_fails)
    first = api_module.create_app(build_services=lambda: fake_services, collection=WORKSPACE)
    with TestClient(first) as client:
        doc_id = upload(client, sample_pdf).json()["document_id"]
        drain(first)
        (doc,) = listed(client)
        assert (doc["status"], doc["chunks"]) == ("failed", 2)  # half written

    monkeypatch.setattr(fake_services.client, "upsert", real_upsert)
    restarted = api_module.create_app(build_services=lambda: fake_services, collection=WORKSPACE)  # job map lost
    with TestClient(restarted) as client:
        (doc,) = listed(client)
        assert (doc["document_id"], doc["status"], doc["chunks"]) == (doc_id, "incomplete", 2)  # never "ready"
        partial_chat = client.post("/api/chat", json={"question": "What was revenue in 2025?"}).json()
        assert partial_chat["abstained"] is True and partial_chat["sources"] == []  # its 2 points are not searched
        assert fake_services.answer_model.calls == []  # nothing was retrieved, so the model was never asked

        again = upload(client, sample_pdf)
        assert again.status_code == 202 and again.json()["status"] == "queued"  # re-ingested, not 200
        drain(restarted)
        (doc,) = listed(client)
        assert (doc["status"], doc["chunks"]) == ("ready", 4)
        answered = client.post("/api/chat", json={"question": "What was revenue in 2025?"}).json()
        assert answered["abstained"] is False and answered["citations"][0]["document_id"] == doc_id


def chunks_for(doc_id, n):
    return [Chunk(f"{doc_id}-p1-{i}", doc_id, f"{doc_id}.pdf", 1, ("S",), "text", f"text {i}") for i in range(n)]


def test_only_one_complete_write_counts_as_complete():
    client = QdrantClient(":memory:")
    ensure_collection(client, WORKSPACE)
    for doc_id in ("done", "short", "mixed"):
        replace_document(client, doc_id, chunks_for(doc_id, 3), [VECTOR] * 3, WORKSPACE)
    client.delete(WORKSPACE, points_selector=models.PointIdsList(points=[point_id("short-p1-2")]), wait=True)  # truncated
    client.set_payload(WORKSPACE, payload={"write_id": "an-interrupted-rewrite"}, points=[point_id("mixed-p1-0")])
    client.upsert(WORKSPACE, [models.PointStruct(  # written before chunk_total/write_id existed
        id=point_id("legacy-p1-0"), vector=VECTOR,
        payload={"document_id": "legacy", "source_name": "legacy.pdf", "page_number": 1, "content_type": "text", "text": "x"},
    )])
    assert {d.document_id: d.complete for d in list_documents(client, WORKSPACE)} == {
        "done": True, "short": False, "mixed": False, "legacy": False}
    client.close()


def test_completeness_fields_do_not_reach_prompts_or_chat_sources(api, sample_pdf):
    ingested(api, sample_pdf)
    api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    prompt = str(api.services.answer_model.calls[-1])
    assert "chunk_total" not in prompt and "write_id" not in prompt  # build_messages reads named fields only
    sources = api.post("/api/chat", json={"question": "What was revenue in 2025?"}).json()["sources"]
    assert sources and all("write_id" not in s and "chunk_total" not in s for s in sources)


def test_delete_in_progress_blocks_uploads_of_that_document(api, sample_pdf, monkeypatch):
    doc_id = ingested(api, sample_pdf)["document_id"]
    entered, release = threading.Event(), threading.Event()
    real_delete = api_module.delete_document

    def slow_delete(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return real_delete(*args, **kwargs)

    monkeypatch.setattr(api_module, "delete_document", slow_delete)
    result = {}
    worker = threading.Thread(target=lambda: result.update(response=api.delete(f"/api/documents/{doc_id}")))
    worker.start()
    assert entered.wait(20)

    blocked = upload(api, sample_pdf)
    assert blocked.status_code == 409 and blocked.json()["detail"] == "This document is being deleted."
    assert api.delete(f"/api/documents/{doc_id}").status_code == 409
    assert listed(api) == []  # hidden while it is being deleted

    release.set()
    worker.join(20)
    assert result["response"].status_code == 200
    assert upload(api, sample_pdf).status_code == 202  # accepted again once the deletion has finished
    api.drain()


def test_duplicate_is_not_reported_ready_when_a_delete_starts_during_the_upload(api, sample_pdf, monkeypatch):
    """The delete begins after the upload's first check but before it answers "already ready"."""
    doc_id = ingested(api, sample_pdf)["document_id"]
    entered, release, result = threading.Event(), threading.Event(), {}
    real_delete, real_list = api_module.delete_document, api_module.list_documents

    def slow_delete(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return real_delete(*args, **kwargs)

    worker = threading.Thread(target=lambda: result.update(response=api.delete(f"/api/documents/{doc_id}")))

    def listing_then_delete_starts(*args, **kwargs):
        documents = real_list(*args, **kwargs)  # still shows the complete document
        worker.start()
        assert entered.wait(20)  # the deletion is now registered and running
        return documents

    monkeypatch.setattr(api_module, "delete_document", slow_delete)
    monkeypatch.setattr(api_module, "list_documents", listing_then_delete_starts)
    response = upload(api, sample_pdf)
    assert response.status_code == 409 and response.json()["detail"] == "This document is being deleted."
    release.set()
    worker.join(20)
    assert result["response"].status_code == 200


def test_queue_holds_one_running_and_one_waiting_upload(api, sample_pdf):
    gated = GatedEmbedder()
    api.services.embed_document = gated
    assert upload(api, sample_pdf).status_code == 202
    assert gated.entered.wait(20)

    other = build_text_pdf(OTHER)  # built once: each build gets a new PDF /ID, hence a new document id
    waiting = upload(api, other, "other.pdf")
    assert waiting.status_code == 202 and waiting.json()["status"] == "queued"
    full = upload(api, build_text_pdf(THIRD), "third.pdf")
    assert full.status_code == 429 and full.json() == {"detail": api_module.QUEUE_FULL}
    assert upload(api, other, "other.pdf").status_code == 409  # a duplicate of a queued upload is 409, not 429

    status = {d["source_name"]: (d["status"], d["ingestion"]["stage"]) for d in listed(api)}
    assert status == {"sample.pdf": ("processing", "embed"), "other.pdf": ("queued", "queued")}  # not "extract"

    gated.release.set()
    api.drain()
    assert {d["status"] for d in listed(api)} == {"ready"}
    assert upload(api, build_text_pdf(THIRD), "third.pdf").status_code == 202  # room again
    api.drain()


def test_chat_returns_503_when_qdrant_fails(api, sample_pdf, monkeypatch):
    ingested(api, sample_pdf)

    def qdrant_down(*args, **kwargs):
        raise ResponseHandlingException(ConnectionError("connection refused, key sk-test-secret"))

    monkeypatch.setattr(api.services.client, "scroll", qdrant_down)
    response = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Vector database unavailable") and "sk-test-secret" not in response.text
    assert api.services.answer_model.calls == []


@pytest.mark.parametrize("provider", ["embed_query", "rerank"])
def test_chat_returns_502_when_the_embedder_or_reranker_fails(api, sample_pdf, provider):
    ingested(api, sample_pdf)

    def fails(*args, **kwargs):
        raise RuntimeError(f"{provider} failed: HTTP 500, key sk-test-secret")

    setattr(api.services, provider, fails)
    response = api.post("/api/chat", json={"question": "What was revenue in 2025?"})
    assert response.status_code == 502
    assert "HTTP 500" in response.json()["detail"] and "sk-test-secret" not in response.text
