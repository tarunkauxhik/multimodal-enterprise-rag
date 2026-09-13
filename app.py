"""Streamlit demo: upload PDFs, ask questions, get answers with [document, Page N] citations.

Run from the repository root so .streamlit/config.toml is used:
    uv run streamlit run app.py

Network: binds to 127.0.0.1 by default (.streamlit/config.toml); override with
STREAMLIT_SERVER_ADDRESS / STREAMLIT_SERVER_PORT. There is no authentication, so on
a VPS keep it on localhost behind a reverse proxy or SSH tunnel.

Documents: each browser session has its own Qdrant collection (rag/session.py).
Refreshing the page starts a new, empty session. Documents ingested with the CLI
(`python -m rag.ingest`) live in the shared "documents" collection and are NOT
visible here.
"""

import time

import streamlit as st

from rag import session
from rag.extract import document_id_for

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

SESSION_KEYS = ("collection", "documents", "failed", "result", "last_touch")
ACTIVITY_REFRESH_SECONDS = 60  # at most one activity write per minute while a session is in use


@st.cache_resource(show_spinner=False)
def get_services() -> session.Services:
    return session.build_services()


def error_text(exc: Exception, services: session.Services) -> str:
    return session.redact(f"{type(exc).__name__}: {exc}", services.secrets)


def reset_session_state() -> None:
    state = st.session_state
    for key in SESSION_KEYS:
        state.pop(key, None)
    state.uploader_key = state.get("uploader_key", 0) + 1  # also clears the uploader widget


def init_state(services: session.Services) -> None:
    state = st.session_state
    state.setdefault("uploader_key", 0)
    if "collection" in state:
        return
    try:
        session.cleanup_inactive_sessions(services.client)
        state.collection = session.start_session(services.client)
    except Exception as exc:
        st.error(f"Cannot reach the vector database. Is Qdrant running? ({error_text(exc, services)})")
        st.stop()
    state.documents = {}  # document_id -> IngestResult
    state.failed = {}  # document_id -> (file name, error message)
    state.result = None  # (question, Answer, hits)
    state.last_touch = time.time()


def keep_alive(services: session.Services) -> None:
    """Refresh the session's last-activity time; start over if cleanup already deleted it."""
    state = st.session_state
    now = time.time()
    if now - state.get("last_touch", 0) < ACTIVITY_REFRESH_SECONDS:
        return
    try:
        alive = session.touch_session(services.client, state.collection, now)
    except Exception as exc:
        st.warning(f"Could not record session activity: {error_text(exc, services)}")
        return
    if alive:
        state.last_touch = now
        return
    reset_session_state()
    st.warning(
        f"This session was inactive for more than {session.SESSION_TTL_SECONDS // 3600} hours, "
        "so its documents were deleted. Please upload them again."
    )
    init_state(services)


def ingest(services: session.Services, name: str, data: bytes) -> None:
    state = st.session_state
    document_id = document_id_for(data)
    if document_id in state.documents or document_id in state.failed:
        return  # already processed in this session (Streamlit reruns the script on every interaction)
    if message := session.upload_size_error(len(data), st.get_option("server.maxUploadSize")):
        state.failed[document_id] = (name, message)
        return
    name = session.unique_source_name(name, (r.source_name for r in state.documents.values()))
    with st.status(f"Processing {name}…", expanded=True) as status:
        try:
            result = session.ingest_upload(services, state.collection, data, name, on_stage=status.write)
        except Exception as exc:
            state.failed[document_id] = (name, error_text(exc, services))
            status.update(label=f"Failed: {name}", state="error", expanded=False)
            return
        status.update(label=f"Ready: {name}", state="complete", expanded=False)
    state.documents[document_id] = result
    state.last_touch = time.time()


def sidebar(services: session.Services) -> None:
    state = st.session_state
    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True, key=f"uploader_{state.uploader_key}"
        )
        st.caption(
            f"Up to {st.get_option('server.maxUploadSize')} MB per file. Documents belong to this browser "
            f"session only: refreshing or closing the page starts a new, empty session, and documents unused "
            f"for {session.SESSION_TTL_SECONDS // 3600} hours are deleted. Files ingested with the "
            "command-line tool are not shown here."
        )
        for upload in uploads or []:
            ingest(services, upload.name, upload.getvalue())

        if not state.documents:
            st.caption("No documents in this session yet.")
        for result in state.documents.values():
            st.markdown(f"**{result.source_name}**")
            details = f"{result.pages} pages · {result.chunks} chunks"
            if result.understood_pages:
                details += f" · M3 on pages {', '.join(map(str, result.understood_pages))}"
            st.caption(details)
            if result.chunks == 0:
                st.warning("No text could be extracted from this document.")
            if result.failed_pages:
                st.warning(
                    f"Visual understanding failed on pages {', '.join(map(str, result.failed_pages))}; "
                    "fast text extraction was used for those pages."
                )

        for name, message in state.failed.values():
            st.error(f"{name}: {message}")
        if state.failed and st.button("Retry failed uploads"):
            state.failed = {}
            st.rerun()

        if (state.documents or state.failed) and st.button("Clear session documents"):
            try:
                session.delete_session(services.client, state.collection)
            except Exception as exc:
                st.error(f"Could not clear documents: {error_text(exc, services)}")
                return
            reset_session_state()
            st.rerun()


def render_result(question: str, answer, hits) -> None:
    st.subheader("Answer")
    st.caption(f"Question: {question}")
    if answer.abstained:
        st.warning(answer.text)
    else:
        st.markdown(answer.text)
    if answer.removed_citations:
        st.caption(f"{len(answer.removed_citations)} citation(s) to sources outside the retrieved context were removed.")

    if answer.citations:
        st.markdown("**Citations**")
        for document, page in answer.citations:
            sections = sorted(
                {" › ".join(s["section_path"]) for s in answer.sources if (s["source_name"], s["page_number"]) == (document, page)}
                - {""}
            )
            st.markdown(f"- **[{document}, Page {page}]**" + (f" — {'; '.join(sections)}" if sections else ""))

    with st.expander(f"Retrieved source chunks ({len(hits)})", expanded=answer.abstained):
        if not hits:
            st.caption("No matching chunks were found in this session's documents.")
        for rank, hit in enumerate(hits, start=1):
            p = hit.payload
            section = " › ".join(p["section_path"])
            st.markdown(f"**{rank}. [{p['source_name']}, Page {p['page_number']}]**" + (f" — {section}" if section else ""))
            st.caption(
                f"{p['content_type']} · rerank {hit.rerank_score:.3f} · RRF {hit.rrf_score:.4f} · "
                f"dense rank {hit.dense_rank or '–'} · BM25 rank {hit.bm25_rank or '–'}"
            )
            st.markdown(p["text"])  # HTML in document text is not rendered
            st.divider()


def ask(services: session.Services) -> None:
    state = st.session_state
    st.title("Document Q&A")
    st.caption("Answers come only from the PDFs uploaded in this browser session, cited as [document, Page N].")
    if not state.documents:
        st.info("Upload one or more PDFs in the sidebar to start asking questions.")
        return

    with st.form("ask"):
        question = st.text_input("Question", key="question", placeholder="e.g. How many days of paid leave do employees get?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your documents and writing the answer…"):
                try:
                    answer, hits = session.answer_question(services, state.collection, question.strip())
                    state.result = (question.strip(), answer, hits)
                    state.last_touch = time.time()
                except Exception as exc:
                    state.result = None
                    st.error(f"Could not answer the question: {error_text(exc, services)}")
    if state.result:
        render_result(*state.result)


def main() -> None:
    try:
        services = get_services()
    except RuntimeError as exc:
        st.error(f"Configuration error: {exc}. Set these in .env or the environment, then restart the app.")
        st.stop()
    init_state(services)
    keep_alive(services)
    sidebar(services)
    ask(services)


main()
