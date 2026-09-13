import time

from qdrant_client import models

from conftest import build_text_pdf
from rag import session
from rag.config import EMBED_DIM, QDRANT_COLLECTION
from rag.extract import document_id_for
from rag.store import iter_payloads

DAY = session.SESSION_TTL_SECONDS
OTHER_PDF_LINES = [
    "Office Handbook",
    "Office hours are 9:30 to 18:00 on weekdays.",
    "Revenue reporting questions go to the finance team.",
]


def test_new_session_collections_are_unique_and_timestamped():
    a, b = session.new_session_collection(now=1_700_000_000), session.new_session_collection(now=1_700_000_000)
    assert a != b and a.startswith("session_1700000000_") and len(a.rsplit("_", 1)[1]) == 32


def test_start_session_creates_collection_with_activity(fake_services):
    collection = session.start_session(fake_services.client, now=1_700_000_123)
    assert fake_services.client.collection_exists(collection)
    assert session.last_activity(fake_services.client, collection) == 1_700_000_123


# --- Inactivity cleanup ----------------------------------------------------------------


def test_cleanup_uses_last_activity_not_creation_time(fake_services):
    client, now = fake_services.client, 1_800_000_000
    old_but_active = session.start_session(client, now=now - 3 * DAY)
    session.touch_session(client, old_but_active, now=now - 60)
    new_but_idle = session.start_session(client, now=now - DAY - 120)  # created recently, then left idle
    idle = session.start_session(client, now=now - 2 * DAY)

    assert sorted(session.cleanup_inactive_sessions(client, now=now)) == sorted([new_but_idle, idle])
    assert client.collection_exists(old_but_active)


def test_cleanup_never_touches_shared_or_unrecognised_collections(fake_services):
    client, now = fake_services.client, 1_800_000_000
    params = models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
    legacy_idle = session.new_session_collection(now=now - 2 * DAY)  # no activity metadata: creation time used
    legacy_fresh = session.new_session_collection(now=now - 60)
    for name in (QDRANT_COLLECTION, "session_not-a-timestamp", legacy_idle, legacy_fresh):
        client.create_collection(name, vectors_config=params)

    assert session.cleanup_inactive_sessions(client, now=now) == [legacy_idle]
    remaining = {c.name for c in client.get_collections().collections}
    assert remaining == {QDRANT_COLLECTION, "session_not-a-timestamp", legacy_fresh}


def test_upload_and_question_refresh_activity(fake_services, sample_pdf):
    client = fake_services.client
    collection = session.start_session(client, now=1_000)

    session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf")
    after_upload = session.last_activity(client, collection)
    assert after_upload >= int(time.time()) - 5

    session.touch_session(client, collection, now=1_000)
    session.answer_question(fake_services, collection, "What was revenue in 2025?")
    assert session.last_activity(client, collection) >= int(time.time()) - 5

    assert session.cleanup_inactive_sessions(client) == []  # recently used: kept


def test_touch_reports_deleted_session(fake_services):
    collection = session.start_session(fake_services.client)
    session.delete_session(fake_services.client, collection)
    assert session.touch_session(fake_services.client, collection) is False
    assert not fake_services.client.collection_exists(collection)  # touching never recreates it


# --- Isolation ------------------------------------------------------------------------


def test_sessions_cannot_retrieve_each_others_documents(fake_services, sample_pdf):
    alice, bob = session.start_session(fake_services.client), session.start_session(fake_services.client)
    alice_doc = session.ingest_upload(fake_services, alice, sample_pdf, "financials.pdf")
    bob_doc = session.ingest_upload(fake_services, bob, build_text_pdf(OTHER_PDF_LINES), "office.pdf")

    # "revenue 2025" matches Alice's table best, but Bob must only ever see his own document.
    _, bob_hits = session.answer_question(fake_services, bob, "What was revenue in 2025?")
    assert bob_hits and {h.payload["document_id"] for h in bob_hits} == {bob_doc.document_id}

    _, alice_hits = session.answer_question(fake_services, alice, "What was revenue in 2025?")
    assert {h.payload["document_id"] for h in alice_hits} == {alice_doc.document_id}

    assert not fake_services.client.collection_exists(QDRANT_COLLECTION)  # nothing leaks into the CLI collection


def test_same_pdf_in_two_sessions_is_independent(fake_services, sample_pdf):
    alice, bob = session.start_session(fake_services.client), session.start_session(fake_services.client)
    session.ingest_upload(fake_services, alice, sample_pdf, "sample.pdf")
    session.ingest_upload(fake_services, bob, sample_pdf, "sample.pdf")

    session.delete_session(fake_services.client, alice)
    assert not fake_services.client.collection_exists(alice)
    answer, hits = session.answer_question(fake_services, bob, "What was revenue in 2025?")
    assert hits and not answer.abstained


def test_cli_ingested_documents_are_not_visible_in_a_session(fake_services, sample_pdf):
    from rag.embed import EmbeddingCache
    from rag.ingest import ingest_pdf

    cache = EmbeddingCache(fake_services.embed_cache_path)
    ingest_pdf(sample_pdf, "cli.pdf", client=fake_services.client, embed_batch=fake_services.embed_document, cache=cache)
    cache.close()

    answer, hits = session.answer_question(fake_services, session.start_session(fake_services.client), "revenue 2025")
    assert fake_services.client.count(QDRANT_COLLECTION).count > 0
    assert hits == [] and answer.abstained


# --- Flow, citations, uploads -----------------------------------------------------------


def test_empty_session_abstains_without_calling_the_model(fake_services):
    answer, hits = session.answer_question(fake_services, session.start_session(fake_services.client), "What was revenue?")
    assert answer.abstained and hits == []
    assert fake_services.answer_model.calls == []


def test_answer_cites_document_and_page_from_session_hits(fake_services, sample_pdf):
    collection = session.start_session(fake_services.client)
    session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf")
    answer, hits = session.answer_question(fake_services, collection, "What was revenue in 2025?")
    assert not answer.abstained and answer.citations == [("sample.pdf", 1)]
    assert answer.text == "Revenue was 120 in 2025 [sample.pdf, Page 1]."
    assert all(s["source_name"] == "sample.pdf" and s["page_number"] == 1 for s in answer.sources)
    prompt = fake_services.answer_model.calls[0][1]["content"]
    assert all(h.payload["text"].split("\n")[0] in prompt for h in hits)


def test_ingest_upload_reports_stages_and_falls_back_when_m3_fails(fake_services, sample_pdf):
    collection, stages = session.start_session(fake_services.client), []
    result = session.ingest_upload(fake_services, collection, sample_pdf, "sample.pdf", on_stage=stages.append)
    assert [s.split(" ")[0] for s in stages] == ["Extracting", "Checking", "Chunking", "Embedding", "Storing"]
    assert result.document_id == document_id_for(sample_pdf)
    assert result.failed_pages == [1] and result.understood_pages == [] and result.chunks > 0
    payloads = list(iter_payloads(fake_services.client, collection))
    assert len(payloads) == result.chunks and {p["source_name"] for p in payloads} == {"sample.pdf"}


def test_unique_source_name_keeps_citations_unambiguous():
    assert session.unique_source_name("report.pdf", []) == "report.pdf"
    assert session.unique_source_name("report.pdf", ["report.pdf"]) == "report (2).pdf"
    assert session.unique_source_name("report.pdf", ["report.pdf", "report (2).pdf"]) == "report (3).pdf"
    assert session.unique_source_name("README", ["README"]) == "README (2)"


def test_upload_size_error():
    assert session.upload_size_error(200 * 1024 * 1024, 200) is None
    message = session.upload_size_error(200 * 1024 * 1024 + 1, 200)
    assert "above the 200 MB upload limit" in message and "STREAMLIT_SERVER_MAX_UPLOAD_SIZE" in message


def test_answer_model_runs_with_thinking_disabled_understanding_keeps_default(monkeypatch):
    for name in ("MINIMAX_API_KEY", "GEMINI_API_KEY", "JINA_API_KEY"):
        monkeypatch.setenv(name, "test-key")
    calls = []
    monkeypatch.setattr(session, "minimax_client", lambda *args, **kwargs: calls.append(kwargs) or (lambda m: ""))
    session.build_services()
    answer_kwargs, understand_kwargs = calls
    assert answer_kwargs["thinking"] is False
    assert "thinking" not in understand_kwargs


def test_redact_removes_every_secret():
    assert session.redact("key sk-a failed, retry sk-b", ["sk-a", "sk-b"]) == "key *** failed, retry ***"
