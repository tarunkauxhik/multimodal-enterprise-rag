"""Team UI (ui.py): an HTTP client of api.py. Offline: fake services, in-process API, AppTest."""

import ast
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException
from streamlit.testing.v1 import AppTest

import api as api_module
import ui
from rag.generate import ABSTAIN_MESSAGE
from tests.conftest import RecordingModel, build_text_pdf, hash_embed

ROOT = Path(__file__).resolve().parent.parent
UI = str(ROOT / "ui.py")
QUESTION = "What was revenue in 2025?"


# --- ApiClient against the real API (in process, fake RAG services) --------------------------


class GatedEmbedder:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, texts):
        self.entered.set()
        assert self.release.wait(20)
        return hash_embed(texts)


@pytest.fixture
def live(fake_services):
    app = api_module.create_app(build_services=lambda: fake_services, collection="documents", max_upload_mb=1)
    with TestClient(app) as http:  # TestClient is an httpx.Client: ApiClient uses it unchanged
        client = ui.ApiClient(http)
        client.services = fake_services
        client.drain = lambda: app.state.ingest_executor.submit(lambda: None).result(timeout=60)
        yield client


def test_client_health_and_empty_workspace(live):
    assert live.health() == {"status": "ok", "qdrant": True, "collection": "documents"}
    assert live.documents() == []


def test_client_upload_202_then_ready_then_duplicate_200(live, sample_pdf):
    status, body = live.upload("sample.pdf", sample_pdf)
    assert status == 202 and body["status"] == "queued"
    live.drain()
    (doc,) = live.documents()
    assert doc["status"] == "ready" and doc["document_id"] == body["document_id"]
    assert live.upload("copy.pdf", sample_pdf) == (200, {"document_id": doc["document_id"], "source_name": "sample.pdf", "status": "ready"})


@pytest.mark.parametrize("data, status", [(b"not a pdf", 415), (b"%PDF-" + b"0" * (1024 * 1024), 413)],
                         ids=["not-a-pdf-415", "too-large-413"])
def test_client_rejected_uploads_raise_with_the_api_message(live, data, status):
    with pytest.raises(ui.ApiError) as raised:
        live.upload("file.pdf", data)
    assert raised.value.status == status and raised.value.detail


def test_client_409_while_processing_and_429_when_the_queue_is_full(live, sample_pdf):
    gated = GatedEmbedder()
    live.services.embed_document = gated
    live.upload("sample.pdf", sample_pdf)
    assert gated.entered.wait(20)
    live.upload("other.pdf", build_text_pdf(["Line one of another report.", "Second line with more words.", "Third line."]))
    for data, status in ((sample_pdf, 409), (build_text_pdf(["A third report.", "Its second line.", "Its third."]), 429)):
        with pytest.raises(ui.ApiError) as raised:
            live.upload("x.pdf", data)
        assert raised.value.status == status
    assert [d["status"] for d in sorted(live.documents(), key=lambda d: d["source_name"])] == ["queued", "processing"]
    gated.release.set()
    live.drain()


def test_client_delete_then_404(live, sample_pdf):
    doc_id = live.upload("sample.pdf", sample_pdf)[1]["document_id"]
    live.drain()
    assert live.delete(doc_id)["deleted_points"] > 0
    with pytest.raises(ui.ApiError) as raised:
        live.delete(doc_id)
    assert raised.value.status == 404


def test_client_chat_answer_citations_and_abstention(live, sample_pdf):
    assert live.chat(QUESTION)["abstained"] is True  # empty workspace
    live.upload("sample.pdf", sample_pdf)
    live.drain()
    reply = live.chat(QUESTION)
    assert set(reply) == set(ui.REPLY_FIELDS) and reply["abstained"] is False
    assert reply["citations"][0]["document"] == "sample.pdf" and reply["sources"][0]["text"]


def test_client_502_and_503(live, sample_pdf, monkeypatch):
    live.upload("sample.pdf", sample_pdf)
    live.drain()
    live.services.answer_model = RecordingModel(RuntimeError("model down, key sk-test-secret"))
    with pytest.raises(ui.ApiError) as raised:
        live.chat(QUESTION)
    assert raised.value.status == 502 and "sk-test-secret" not in raised.value.detail

    def qdrant_down(*args, **kwargs):
        raise ResponseHandlingException(ConnectionError("refused"))

    monkeypatch.setattr(live.services.client, "scroll", qdrant_down)
    with pytest.raises(ui.ApiError) as raised:
        live.chat(QUESTION)
    assert raised.value.status == 503


def test_client_reports_an_unreachable_api_as_status_0():
    def refused(request):
        raise httpx.ConnectError("connection refused")

    client = ui.ApiClient(httpx.Client(base_url="http://127.0.0.1:9", transport=httpx.MockTransport(refused)))
    with pytest.raises(ui.ApiError) as raised:
        client.health()
    assert raised.value.status == 0


# --- wording and upload bookkeeping (pure) ---------------------------------------------------


def doc(status, stage=None, error=None, **extra):
    ingestion = None if stage is None else {"stage": stage, "pages": 2, "chunks": 4, "new_embeddings": 3, "error": error}
    return {"document_id": f"{status:0<16}"[:16], "source_name": f"{status}.pdf", "status": status, "chunks": 4,
            "pages_with_chunks": 2, "content_types": {"text": 4}, "ingestion": ingestion} | extra


EVERY_STATUS = [doc("ready"), doc("queued", "queued"), doc("processing", "embed"), doc("failed", "extract", "ValueError"),
                doc("incomplete"), doc("empty", "done")]


def test_every_status_has_a_label_badge_and_plain_description():
    text = {d["status"]: ui.describe(d) for d in EVERY_STATUS}
    assert set(ui.STATUS) == set(text)  # all six API statuses are covered
    assert text["ready"] == "4 passages · 2 pages"
    assert text["queued"] == "Waiting for the current upload to finish."
    assert text["processing"] == "Indexing passages… 3 passages so far."
    assert "could not be read as a PDF" in text["failed"] and "ValueError" not in text["failed"]
    assert "Upload the same PDF again to repair it" in text["incomplete"]
    assert "No text could be extracted" in text["empty"]


def test_failures_never_show_exception_class_names():
    for stage in [*ui.FAILURES, "unknown"]:
        assert "Error" not in ui.describe(doc("failed", stage, "RuntimeError"))


def test_polling_only_while_something_is_queued_or_processing():
    assert ui.is_active([doc("queued", "queued")]) and ui.is_active([doc("processing", "extract")])
    assert not ui.is_active([doc("ready"), doc("failed", "embed"), doc("incomplete"), doc("empty", "done")])
    assert not ui.is_active([])


class FakeApi:
    """A scripted ApiClient for the UI tests."""

    def __init__(self, docs=(), health="ok", reply=None, upload=(202, None), errors=None):
        self.docs, self.health_status, self.reply, self.upload_result = list(docs), health, reply, upload
        self.errors = errors or {}
        self.calls = []

    def _maybe_fail(self, name):
        self.calls.append(name)
        if name in self.errors:
            raise self.errors[name]

    def health(self):
        self._maybe_fail("health")
        return {"status": self.health_status, "qdrant": self.health_status == "ok", "collection": "documents"}

    def documents(self):
        self._maybe_fail("documents")
        return self.docs

    def upload(self, name, data):
        self._maybe_fail("upload")
        status, body = self.upload_result
        return status, body or {"document_id": "0" * 16, "source_name": name, "status": "queued" if status == 202 else "ready"}

    def delete(self, document_id):
        self._maybe_fail(("delete", document_id))
        return {"document_id": document_id, "deleted_points": 4}

    def chat(self, question):
        self._maybe_fail(("chat", question))
        return self.reply


@pytest.mark.parametrize("upload, error, kind, text", [
    ((202, None), None, "success", "a.pdf was added and is being processed."),
    ((200, None), None, "info", "a.pdf is already in the workspace."),
    (None, ui.ApiError(409, "This document is already queued or being processed."), "error", "already queued"),
    (None, ui.ApiError(429, api_module.QUEUE_FULL), "error", "queue is full"),
    (None, ui.ApiError(413, "File is larger than the 200 MB upload limit."), "error", "200 MB upload limit"),
    (None, ui.ApiError(415, "Only PDF files are accepted."), "error", "Only PDF files"),
    (None, ui.ApiError(503, "Vector database unavailable: ConnectError"), "error", "index is unavailable"),
    (None, ui.ApiError(0), "error", "Can't reach the document service"),
])
def test_upload_outcomes_become_friendly_notices(upload, error, kind, text):
    fake = FakeApi(upload=upload or (202, None), errors={"upload": error} if error else None)
    notice = ui.upload_file(fake, "a.pdf", b"%PDF-1.7 a", set())
    assert notice[0] == kind and text in notice[1].replace("\\", "")  # names are Markdown-escaped
    assert "ConnectError" not in notice[1]


def test_file_names_in_notices_cannot_render_markdown():
    notice = ui.upload_file(FakeApi(), "![x](evil) **big**.pdf", b"%PDF- b", set())
    assert "![" not in notice[1] and "**" not in notice[1] and "\\!\\[x\\]" in notice[1]


def test_accepted_files_are_posted_once_and_rejected_ones_can_be_retried():
    sent = set()
    fake = FakeApi(errors={"upload": ui.ApiError(429, api_module.QUEUE_FULL)})
    assert ui.upload_file(fake, "a.pdf", b"%PDF- a", sent)[0] == "error"
    del fake.errors["upload"]
    assert ui.upload_file(fake, "a.pdf", b"%PDF- a", sent)[0] == "success"  # the 429 was not remembered
    assert ui.upload_file(fake, "a.pdf", b"%PDF- a", sent) is None  # a rerun does not POST again
    assert fake.calls == ["upload", "upload"] and all(len(d) == 64 for d in sent)  # hashes, not bytes


def test_plain_shows_markdown_and_html_literally():
    escaped = ui.plain("**bold** [link](http://x) ![img](http://x/i.png) <b>x</b> :material/home: $x$")
    for raw in ("**", "](", "![", "<b>", "$x$", "http:"):
        assert raw not in escaped  # no emphasis, link, image, HTML, maths or autolink can form
    assert "\\:material/home\\:" in escaped


def test_ui_never_imports_rag_or_renders_unsafe_html():
    source = (ROOT / "ui.py").read_text(encoding="utf-8")
    imported = [n.module or "" for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if m == "rag" or m.startswith("rag.")]
    for forbidden in ("unsafe_allow_html", "st.html", "raw_output", "write_id", "chunk_total", "rerank_score", "dense_rank"):
        assert forbidden not in source


# --- the page (AppTest with a fake client) ------------------------------------------------------


def run_ui(fake, **session):
    at = AppTest.from_file(UI, default_timeout=30)
    at.session_state["api_client"] = fake
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


def texts(elements):
    return " ".join(e.value for e in elements)


def test_unreachable_api_shows_how_to_start_it(monkeypatch):
    at = run_ui(FakeApi(errors={"health": ui.ApiError(0)}))
    assert not at.exception
    assert "Can't reach the document service" in texts(at.error)
    assert f"`{ui.API_URL}`" in texts(at.error)  # shown as code: the server's localhost is no link for the browser
    assert at.code[0].value == ui.START_API and "--workers 1" in ui.START_API
    assert not at.chat_input  # nothing else is drawn


def test_qdrant_outage_pauses_uploads_and_questions():
    fake = FakeApi(health="unavailable")
    at = run_ui(fake)
    assert not at.exception and "document index is unavailable" in texts(at.warning)
    assert at.chat_input[0].disabled and "documents" not in fake.calls


def test_empty_workspace_explains_how_to_start():
    at = run_ui(FakeApi())
    assert "Add a PDF in the sidebar to start" in texts(at.info)
    assert "No documents yet" in texts(at.sidebar.caption)
    assert at.chat_input[0].disabled


def test_every_status_is_listed_with_a_plain_description():
    at = run_ui(FakeApi(EVERY_STATUS))
    assert not at.exception
    names = texts(at.sidebar.markdown)
    captions = texts(at.sidebar.caption)
    for d in EVERY_STATUS:
        assert d["source_name"].split(".")[0] in names and ui.describe(d) in captions
        assert ui.STATUS[d["status"]][0] in names  # the badge label
    assert "1 of 6 ready" in captions and "ValueError" not in names + captions
    assert not at.chat_input[0].disabled  # one document is ready
    deletes = {b.key: b.disabled for b in at.sidebar.button if b.key.startswith("delete_")}
    assert deletes[f"delete_{doc('queued')['document_id']}"] and not deletes[f"delete_{doc('ready')['document_id']}"]


def test_processing_only_workspace_waits_for_a_ready_document():
    at = run_ui(FakeApi([doc("processing", "extract")]))
    assert "still being processed" in texts(at.info) and at.chat_input[0].disabled


def test_documents_failing_to_load_stop_questions_with_a_retry():
    at = run_ui(FakeApi(errors={"documents": ui.ApiError(503, "Vector database unavailable")}))
    assert "index is unavailable" in texts(at.sidebar.warning) and at.chat_input[0].disabled
    assert any(b.key == "retry_documents" for b in at.sidebar.button)


def test_retry_after_a_loading_failure_re_enables_questions():
    fake = FakeApi([doc("ready")], errors={"documents": ui.ApiError(0)})
    at = run_ui(fake)
    assert at.chat_input[0].disabled and "could not be loaded" in texts(at.warning)
    del fake.errors["documents"]  # the API is back
    at.sidebar.button(key="retry_documents").click().run()
    assert not at.chat_input[0].disabled and not at.warning  # the whole page recovered, not just the list


def test_delete_asks_for_confirmation_first():
    ready = doc("ready")
    fake = FakeApi([ready])
    at = run_ui(fake)
    at.sidebar.button(key=f"delete_{ready['document_id']}").click().run()
    assert "Delete this document for everyone?" in texts(at.sidebar.warning)
    assert ("delete", ready["document_id"]) not in fake.calls  # nothing deleted yet

    at.sidebar.button(key=f"cancel_{ready['document_id']}").click().run()
    assert "Delete this document for everyone?" not in texts(at.sidebar.warning)

    at.sidebar.button(key=f"delete_{ready['document_id']}").click().run()
    at.sidebar.button(key=f"confirm_{ready['document_id']}").click().run()
    assert ("delete", ready["document_id"]) in fake.calls
    assert "ready.pdf was deleted." in texts(at.sidebar.success).replace("\\", "")


def test_upload_notices_are_shown_once():
    at = run_ui(FakeApi([doc("queued", "queued")]), notices=[("success", "a.pdf was added and is being processed."),
                                                              ("error", api_module.QUEUE_FULL)])
    assert "a.pdf was added" in texts(at.sidebar.success) and "queue is full" in texts(at.sidebar.error)
    at.run()
    assert not at.sidebar.success and not at.sidebar.error


SNIPPET = "Revenue grew. **not bold** <b>not html</b> [not a link](http://x) ![no image](http://x/i.png)"
REPLY = {
    "answer": "Revenue was 120 in 2025 [sample.pdf, Page 1].",
    "abstained": False,
    "citations": [{"document": "sample.pdf", "document_id": "a" * 16, "page": 1}],
    "sources": [{"chunk_id": "c1", "document_id": "a" * 16, "source_name": "sample.pdf", "page_number": 1,
                 "section_path": ["Annual Report", "2. Financials"], "content_type": "table", "text": SNIPPET}],
}


def ask(at, question=QUESTION):
    return at.chat_input[0].set_value(question).run()


def test_chat_shows_the_answer_citations_and_plain_text_sources():
    fake = FakeApi([doc("ready")], reply=REPLY)
    at = ask(run_ui(fake))
    assert not at.exception and ("chat", QUESTION) in fake.calls
    rendered = texts(at.markdown)
    assert "Revenue was 120 in 2025" in rendered.replace("\\", "")
    assert "sample.pdf" in rendered.replace("\\", "") and "Page 1" in rendered
    assert at.text[0].value == SNIPPET  # the source snippet, exactly and literally
    assert "Annual Report › 2" in texts(at.caption).replace("\\", "")
    assert "Each question is answered on its own" in texts(at.caption)


def test_chat_history_is_kept_and_each_question_is_its_own_request():
    fake = FakeApi([doc("ready")], reply=REPLY)
    at = ask(ask(run_ui(fake)), "And the profit?")
    assert [c for c in fake.calls if isinstance(c, tuple)] == [("chat", QUESTION), ("chat", "And the profit?")]
    assert len(at.chat_message) == 4 and len(at.session_state["messages"]) == 4
    assert all(set(m) <= {"role", "text", *ui.REPLY_FIELDS, "error"} for m in at.session_state["messages"])


def test_abstention_is_neutral_information_not_an_error():
    at = ask(run_ui(FakeApi([doc("ready")], reply={"answer": ABSTAIN_MESSAGE, "abstained": True, "citations": [], "sources": []})))
    assert ABSTAIN_MESSAGE in texts(at.info).replace("\\", "") and not at.error


@pytest.mark.parametrize("status, text", [(502, "answer service did not respond"), (503, "index is unavailable")])
def test_chat_failures_are_friendly(status, text):
    fake = FakeApi([doc("ready")], errors={("chat", QUESTION): ui.ApiError(status, "RuntimeError: sk-test-secret")})
    at = ask(run_ui(fake))
    assert text in texts(at.error) and "RuntimeError" not in texts(at.error) and "sk-test-secret" not in texts(at.error)


def test_unexpected_reply_fields_are_never_kept_or_shown():
    leaky = REPLY | {"raw_output": "<think>PRIVATE-REASONING</think>", "debug": {"dense_rank": 1}}
    at = ask(run_ui(FakeApi([doc("ready")], reply=leaky)))
    shown = texts(at.markdown) + texts(at.text) + texts(at.caption)
    assert "PRIVATE-REASONING" not in shown and "dense_rank" not in shown
    assert "raw_output" not in at.session_state["messages"][-1]
