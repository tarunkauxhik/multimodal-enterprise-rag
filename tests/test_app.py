"""Streamlit UI tests with AppTest and fake services (offline).

AppTest cannot drive st.file_uploader, so uploads are ingested through
rag.session directly and placed into the session state the app uses.
"""

import tomllib
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag import session

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def fresh_resource_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app(monkeypatch, services, state=None):
    monkeypatch.setattr(session, "build_services", lambda: services)
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return at.run()


def session_with_sample(services, sample_pdf):
    collection = session.start_session(services.client)
    result = session.ingest_upload(services, collection, sample_pdf, "sample.pdf")
    return {"collection": collection, "documents": {result.document_id: result}, "failed": {}, "result": None}


def ask(at, question):
    at.text_input(key="question").input(question)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def test_streamlit_config_binds_to_localhost_with_upload_limit():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["maxUploadSize"] == 200


def test_missing_configuration_shows_clear_error(monkeypatch):
    def missing():
        raise RuntimeError("Missing required environment variables: GEMINI_API_KEY")

    monkeypatch.setattr(session, "build_services", missing)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "GEMINI_API_KEY" in at.error[0].value


def test_fresh_session_gets_private_collection_and_explains_lifetime(monkeypatch, fake_services):
    at = run_app(monkeypatch, fake_services)
    assert not at.exception
    assert at.title[0].value == "Document Q&A"
    assert any("Upload one or more PDFs" in info.value for info in at.info)
    collection = at.session_state["collection"]
    assert collection.startswith(session.SESSION_PREFIX) and fake_services.client.collection_exists(collection)
    assert session.last_activity(fake_services.client, collection) is not None
    notes = " ".join(c.value for c in at.sidebar.caption)
    assert "200 MB" in notes and "refreshing or closing the page starts a new, empty session" in notes
    assert "command-line tool are not shown here" in notes


def test_two_app_sessions_get_different_collections(monkeypatch, fake_services):
    first = run_app(monkeypatch, fake_services)
    second = run_app(monkeypatch, fake_services)
    assert first.session_state["collection"] != second.session_state["collection"]


def test_question_shows_answer_document_page_citations_and_chunks(monkeypatch, fake_services, sample_pdf):
    at = run_app(monkeypatch, fake_services, session_with_sample(fake_services, sample_pdf))
    assert any("sample.pdf" in md.value for md in at.sidebar.markdown)

    ask(at, "What was revenue in 2025?")
    assert not at.exception and not at.error
    markdown = [md.value for md in at.markdown]
    assert "Revenue was 120 in 2025 [sample.pdf, Page 1]." in markdown
    assert any(md.startswith("- **[sample.pdf, Page 1]**") for md in markdown)
    assert at.expander[0].label.startswith("Retrieved source chunks (")
    assert any("[sample.pdf, Page 1]**" in md and md.startswith("**1.") for md in markdown)


def test_using_the_app_refreshes_session_activity(monkeypatch, fake_services, sample_pdf):
    state = session_with_sample(fake_services, sample_pdf)
    session.touch_session(fake_services.client, state["collection"], now=1_000)  # looks idle
    run_app(monkeypatch, fake_services, state)  # no last_touch yet: the app records activity on this run
    assert session.last_activity(fake_services.client, state["collection"]) > 1_000


def test_expired_session_starts_over_with_a_notice(monkeypatch, fake_services, sample_pdf):
    state = session_with_sample(fake_services, sample_pdf)
    session.delete_session(fake_services.client, state["collection"])  # as inactivity cleanup would

    at = run_app(monkeypatch, fake_services, state)
    assert not at.exception
    assert any("inactive for more than 24 hours" in w.value for w in at.warning)
    new_collection = at.session_state["collection"]
    assert new_collection != state["collection"] and fake_services.client.collection_exists(new_collection)
    assert at.session_state["documents"] == {}


def test_answer_errors_are_shown_without_secrets(monkeypatch, fake_services, sample_pdf):
    state = session_with_sample(fake_services, sample_pdf)

    def broken(messages):
        raise RuntimeError("MiniMax completion failed: HTTP 401 invalid key sk-test-secret")

    fake_services.answer_model = broken
    at = ask(run_app(monkeypatch, fake_services, state), "What was revenue in 2025?")

    assert not at.exception
    (error,) = at.error
    assert "Could not answer the question" in error.value and "HTTP 401" in error.value
    assert "sk-test-secret" not in error.value
