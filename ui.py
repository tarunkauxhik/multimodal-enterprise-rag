"""Document Q&A: the team workspace UI, an HTTP client of the RAG API (api.py).

    uv run uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1   # start the API first
    uv run streamlit run ui.py                                        # then this UI

The API is the only application boundary: this file never imports rag.*. It calls
RAG_API_URL (default http://127.0.0.1:8000) from the Streamlit server process, so the
API stays private on localhost and is never exposed through the reverse proxy.

The workspace is shared: everyone who can open this UI sees, and can delete, the same
documents. Each question is a separate single-turn API request; the chat history is only
kept for display. Document text and model answers are untrusted: source snippets are
shown as plain text, and answers with Markdown escaped, so nothing a document contains
can turn into a link, an image or HTML.
"""

import hashlib
import os
import re

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
START_API = f"uv run uvicorn api:app --host 127.0.0.1 --port {httpx.URL(API_URL).port or 8000} --workers 1"
POLL_SECONDS = 2
SNIPPET_CHARS = 600
ACTIVE = frozenset({"queued", "processing"})
REPLY_FIELDS = ("answer", "abstained", "citations", "sources")  # the whole chat contract; nothing else is kept

# --- API client -------------------------------------------------------------------------


class ApiError(Exception):
    """The API answered with an error status, or could not be reached at all (status 0)."""

    def __init__(self, status: int, detail: str = ""):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class ApiClient:
    """The five API endpoints. Returns decoded JSON, or raises ApiError."""

    def __init__(self, http: httpx.Client):
        self.http = http

    def _request(self, method: str, path: str, ok: tuple[int, ...] = (200,), **kwargs) -> tuple[int, dict]:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # refused, timed out, reset
            raise ApiError(0) from exc
        if response.status_code not in ok:
            raise ApiError(response.status_code, _detail(response))
        return response.status_code, response.json()

    def health(self) -> dict:
        return self._request("GET", "/api/health", ok=(200, 503))[1]

    def documents(self) -> list[dict]:
        return self._request("GET", "/api/documents")[1]["documents"]

    def upload(self, name: str, data: bytes) -> tuple[int, dict]:
        return self._request("POST", "/api/documents", ok=(200, 202), files={"file": (name, data, "application/pdf")})

    def delete(self, document_id: str) -> dict:
        return self._request("DELETE", f"/api/documents/{document_id}")[1]

    def chat(self, question: str) -> dict:
        return self._request("POST", "/api/chat", json={"question": question})[1]


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        return ""
    return detail if isinstance(detail, str) else ""  # 422 details are lists; not shown


@st.cache_resource(show_spinner=False)
def shared_client() -> ApiClient:
    # Chat can take tens of seconds (retrieval, reranking, generation); everything else is quick.
    return ApiClient(httpx.Client(base_url=API_URL, timeout=httpx.Timeout(30.0, read=180.0)))


def api() -> ApiClient:
    return st.session_state.get("api_client") or shared_client()  # tests put a fake client in session state


# --- wording ------------------------------------------------------------------------------

STATUS = {  # label, badge colour, icon: colour is never the only cue
    "ready": ("Ready", "green", ":material/check_circle:"),
    "queued": ("Queued", "gray", ":material/schedule:"),
    "processing": ("Processing", "blue", ":material/autorenew:"),
    "failed": ("Failed", "red", ":material/error:"),
    "incomplete": ("Incomplete", "orange", ":material/warning:"),
    "empty": ("No text found", "gray", ":material/text_fields:"),
}
STAGES = {
    "extract": "Reading the PDF",
    "understand": "Reading figures, tables and scanned pages",
    "chunk": "Splitting it into passages",
    "embed": "Indexing passages",
    "store": "Saving to the index",
    "setup": "Starting",
    "finish": "Finishing",
    "done": "Finishing",
}
FAILURES = {  # by the stage that failed; exception class names are never shown
    "extract": "This file could not be read as a PDF (it may be damaged or password-protected).",
    "understand": "Reading its pages failed. Upload it again to retry.",
    "chunk": "Preparing its text failed. Upload it again to retry.",
    "embed": "Indexing failed because the embedding service did not respond. Upload it again to retry.",
    "store": "Saving it to the index failed. Upload it again to retry.",
    "setup": "The server could not start processing it. Upload it again to retry.",
    "finish": "It was saved, but processing did not finish cleanly. Upload it again to confirm.",
}
USER_MESSAGES = {  # used when the API sends no message of its own
    409: "This document is already queued, being processed or being deleted.",
    413: "This PDF is larger than the upload limit.",
    415: "Only PDF files are accepted.",
    429: "One PDF is being processed and one is waiting. Try again when the current upload has finished.",
}

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$&:])")


def plain(text: str) -> str:
    """Untrusted text for st.markdown, shown literally: no links, images, HTML or formatting."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def describe(doc: dict) -> str:
    """One plain-language line for the document list."""
    status = doc["status"]
    ingestion = doc.get("ingestion") or {}
    if status == "ready":
        return f"{doc['chunks']} passages · {doc['pages_with_chunks']} pages"
    if status == "queued":
        return "Waiting for the current upload to finish."
    if status == "processing":
        stage = STAGES.get(ingestion.get("stage", ""), "Processing")
        indexed = ingestion.get("new_embeddings", 0)
        return f"{stage}…" + (f" {indexed} passages so far." if ingestion.get("stage") == "embed" and indexed else "")
    if status == "failed":
        return FAILURES.get(ingestion.get("stage", ""), "Processing failed. Upload it again to retry.")
    if status == "incomplete":
        return "Only partly saved, so it is not used for answers. Upload the same PDF again to repair it."
    if status == "empty":
        return "No text could be extracted, so it cannot be used for answers."
    return ""


def friendly(error: ApiError, action: str) -> str:
    if error.status == 0:  # the address as code: it is the server's own localhost, not a link for the browser
        return f"Can't reach the document service at `{API_URL}`."
    if error.status in USER_MESSAGES:  # the API's own messages for these are written for users
        return error.detail or USER_MESSAGES[error.status]
    if error.status == 404:
        return "That document no longer exists."
    if error.status == 503:
        return "The document index is unavailable right now. Try again in a moment."
    if error.status == 502:
        return "The answer service did not respond. Try again in a moment."
    return f"Could not {action} (error {error.status}). Try again."


def is_active(docs: list[dict]) -> bool:
    return any(doc["status"] in ACTIVE for doc in docs)


def upload_file(client: ApiClient, name: str, data: bytes, sent: set[str]) -> tuple[str, str] | None:
    """POST one PDF unless this exact file was already accepted this session; return a notice.

    Only the SHA-256 of accepted files is remembered (Streamlit reruns the script on every
    interaction), never their bytes. Rejected uploads are not remembered, so they can be retried.
    """
    digest = hashlib.sha256(data).hexdigest()
    if digest in sent:
        return None
    try:
        status, body = client.upload(name, data)
    except ApiError as exc:
        return "error", f"{plain(name)}: {friendly(exc, 'upload the file')}"
    sent.add(digest)
    if status == 202:
        return "success", f"{plain(body['source_name'])} was added and is being processed."
    return "info", f"{plain(body['source_name'])} is already in the workspace."


# --- sidebar: workspace -------------------------------------------------------------------

state = st.session_state


def init_state() -> None:
    for key, value in (("messages", []), ("notices", []), ("sent", set()), ("docs", []),
                       ("docs_error", None), ("uploader_key", 0), ("pending_delete", None)):
        if key not in state:
            state[key] = value


NOTICE = {"success": (st.success, ":material/check_circle:"), "info": (st.info, ":material/info:"),
          "error": (st.error, ":material/error:")}


def workspace(client: ApiClient, available: bool) -> None:
    with st.sidebar:
        st.header("Workspace")
        st.caption("Shared: everyone with access sees, and can delete, these documents.")
        files = st.file_uploader(
            "Add PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader_{state.uploader_key}",
            disabled=not available, help="One PDF is processed at a time; one more can wait in the queue.",
        )
        if files:
            state.notices = [n for f in files if (n := upload_file(client, f.name, f.getvalue(), state.sent))]
            state.uploader_key += 1  # a fresh, empty uploader: the files' bytes are released
            st.rerun()
        for kind, text in state.notices:  # cleared at the end of main(), once this run has completed
            show, icon = NOTICE[kind]
            show(text, icon=icon)

        if not available:
            st.caption("Documents are unavailable while the index is down.")
            return
        polling = is_active(state.docs)
        st.fragment(document_list, run_every=POLL_SECONDS if polling else None)(client, polling)


def document_list(client: ApiClient, polling: bool) -> None:
    """Re-runs on its own every POLL_SECONDS while an upload is queued or processing, and only then."""
    recovering = state.docs_error is not None
    try:
        docs = client.documents()
    except ApiError as exc:
        state.docs, state.docs_error = [], friendly(exc, "load the documents")
        st.warning(state.docs_error, icon=":material/cloud_off:")
        st.button("Retry", icon=":material/refresh:", key="retry_documents")
        if polling:
            st.rerun()  # stop polling: the next run registers this list without a timer
        return
    state.docs, state.docs_error = docs, None
    if recovering or is_active(docs) != polling:
        st.rerun()  # loading recovered, or an upload started or finished: refresh the chat and (un)schedule polling

    ready = sum(doc["status"] == "ready" for doc in docs)
    st.subheader("Documents")
    if not docs:
        st.caption("No documents yet. Add a PDF above.")
        return
    st.caption(f"{ready} of {len(docs)} ready to answer questions")
    for doc in docs:
        document_row(client, doc)


def document_row(client: ApiClient, doc: dict) -> None:
    label, colour, icon = STATUS.get(doc["status"], (doc["status"].title(), "gray", ":material/help:"))
    with st.container(border=True):
        st.markdown(f"**{plain(doc['source_name'])}**")
        st.badge(label, icon=icon, color=colour)
        st.caption(describe(doc))
        if state.pending_delete == doc["document_id"]:
            st.warning("Delete this document for everyone? It will no longer be used for answers.", icon=":material/delete:")
            confirm, cancel = st.columns(2)
            if confirm.button("Delete", key=f"confirm_{doc['document_id']}", type="primary", icon=":material/delete:"):
                try:
                    client.delete(doc["document_id"])
                    state.notices = [("success", f"{plain(doc['source_name'])} was deleted.")]  # another user's file name
                except ApiError as exc:
                    state.notices = [("error", friendly(exc, "delete the document"))]
                state.pending_delete = None
                st.rerun()
            cancel.button("Cancel", key=f"cancel_{doc['document_id']}", on_click=set_pending_delete, args=(None,))
        else:
            busy = doc["status"] in ACTIVE
            st.button("Delete", key=f"delete_{doc['document_id']}", type="tertiary", icon=":material/delete:",
                      disabled=busy, help="Available once processing has finished." if busy else None,
                      on_click=set_pending_delete, args=(doc["document_id"],))


def set_pending_delete(document_id: str | None) -> None:
    state.pending_delete = document_id  # a callback: set before the rerun, so the row redraws at once


# --- main: chat ---------------------------------------------------------------------------


def render_reply(reply: dict) -> None:
    if "error" in reply:
        st.error(reply["error"], icon=":material/error:")
        return
    if reply["abstained"]:  # a normal outcome, not a failure
        st.info(plain(reply["answer"]), icon=":material/info:")
        return
    st.markdown(plain(reply["answer"]))
    st.caption("Sources")
    for citation in reply["citations"]:
        chunks = [s for s in reply["sources"]
                  if s["document_id"] == citation["document_id"] and s["page_number"] == citation["page"]]
        with st.container(border=True):
            st.markdown(f":material/description: **{plain(citation['document'])}** · Page {citation['page']}")
            details = sorted({" › ".join(c["section_path"]) for c in chunks} - {""})
            if details:
                st.caption(plain("; ".join(details)))
            text = "\n\n".join(c["text"] for c in chunks)
            st.text(text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "…")  # never rendered


def conversation(client: ApiClient, available: bool) -> None:
    st.title("Document Q&A")
    st.caption("Answers come only from the ready documents in this workspace, with page citations. "
               "Each question is answered on its own: earlier questions are not used as context.")
    ready = [doc for doc in state.docs if doc["status"] == "ready"]

    if not available:
        st.warning("The document index is unavailable, so uploads and questions are paused. "
                   "Try again in a moment.", icon=":material/cloud_off:")
        st.button("Retry", icon=":material/refresh:", key="retry_health")
    elif state.docs_error:
        st.warning("Documents could not be loaded, so questions are paused.", icon=":material/cloud_off:")
    elif not ready and is_active(state.docs):
        st.info("Your documents are still being processed. You can ask questions as soon as one is ready.",
                icon=":material/hourglass_top:")
    elif not ready:
        st.info("Add a PDF in the sidebar to start. Only documents marked Ready are used for answers.",
                icon=":material/upload_file:")

    for message in state.messages:
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/description:"):
            if message["role"] == "user":
                st.markdown(plain(message["text"]))
            else:
                render_reply(message)

    if state.messages and st.button("Clear conversation", type="tertiary", icon=":material/delete_sweep:"):
        state.messages = []
        st.rerun()

    can_ask = available and not state.docs_error and bool(ready)
    question = st.chat_input(
        "Ask a question about your documents" if can_ask else "Questions are available once a document is ready",
        disabled=not can_ask, max_chars=2000,
    )
    if question and question.strip():
        ask(client, question.strip())


def ask(client: ApiClient, question: str) -> None:
    state.messages.append({"role": "user", "text": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(plain(question))
    with st.chat_message("assistant", avatar=":material/description:"):
        with st.spinner("Searching your documents and writing an answer…"):
            try:
                response = client.chat(question)
                reply = {field: response[field] for field in REPLY_FIELDS}
            except ApiError as exc:
                reply = {"error": friendly(exc, "answer the question")}
        render_reply(reply)
    state.messages.append({"role": "assistant", **reply})


def unreachable() -> None:
    st.title("Document Q&A")
    st.error(friendly(ApiError(0), "reach the API"), icon=":material/cloud_off:")
    st.markdown("Start it on this server, then retry:")
    st.code(START_API, language="bash")
    st.button("Retry", icon=":material/refresh:", key="retry_api")


def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon=":material/description:", layout="centered")
    init_state()
    client = api()
    try:
        health = client.health()
    except ApiError:
        unreachable()
        return
    available = health.get("status") == "ok"
    workspace(client, available)  # also loads state.docs, before the chat needs them
    conversation(client, available)
    # Notices are shown until a run completes. A rerun the document list forces (to start or stop
    # polling) aborts the run before this line, so they are still there in the run the user sees.
    state.notices = []


if __name__ == "__main__":
    # Run through the importable module, so the app and anything importing `ui` (the tests) share one set
    # of classes: ApiError raised by a client built from `ui` is then caught here too.
    import ui

    ui.main()
