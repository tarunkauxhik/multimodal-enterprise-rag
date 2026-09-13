# Multimodal Enterprise RAG

GitHub is the source of truth. The VPS is deployment/runtime only (runs Qdrant and the app).

## Pipeline (benchmarked — do not replace without explaining why first)

PDF
→ PyMuPDF4LLM extraction
→ selective MiniMax M3 multimodal understanding (only pages that need it)
→ structure-aware chunking
→ Gemini Embedding 2
→ Qdrant (dense) + separate rank-bm25 index
→ application-level RRF
→ Jina reranking
→ MiniMax M3 grounded answer + citations

| Decision | Value |
|---|---|
| LLM (understanding + answers) | `MiniMax-M3`, OpenAI-compatible API, base URL via `MINIMAX_BASE_URL` |
| Embeddings | `gemini-embedding-2`, 768 dims, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` |
| Vector store | Qdrant, cosine, 768 dims |
| Sparse | `rank-bm25`, separate local/in-memory index (V1) |
| Fusion | RRF in app code, k=60; dense top 20 + BM25 top 20 → top 10 |
| Rerank | `jina-reranker-v3`, RRF top 10 → top 5 |
| Generation | MiniMax-M3 over the top 5 chunks, thinking disabled (`GENERATION_THINKING`; see MiniMax-AI/MiniMax-M3#28) |
| Citations | `[document.pdf, Page X]` (validated against supplied chunks); bare `[Page X]` accepted only when exactly one supplied document has that page |

All of these live as constants in `rag/config.py`.

## Rules

- V1 stack: Python 3.12, Streamlit, Qdrant. No FastAPI, Next.js, LangChain, LangGraph, Redis, Kubernetes, or extra framework layers.
- Add a dependency only in the step that needs it; prefer stdlib or already-installed packages.
- API keys come only from environment variables (`rag.config.load_settings`). Never hardcode, print, or log them.
- Never commit PDFs, databases, caches, model files, BM25 index files, or `.env`. Runtime data lives under `data/` (gitignored).
- One module per pipeline stage in `rag/`; no interfaces with a single implementation.
- Non-trivial logic gets one small test in `tests/`.
- Git: commits and pushes are authored and committed solely by `tarunkauxhik`. Never add `Co-Authored-By`, "Generated with", session links, or any other AI attribution to commits, PRs, or code. Never change `git config user.*`.

## Commands

```
uv sync                                   # create .venv from pyproject/uv.lock
uv run pytest                             # tests (offline; in-memory Qdrant, fake embedder)
uv run python -m rag.ingest file.pdf ...  # extract -> chunk -> embed -> Qdrant (idempotent)
uv run python -m rag.retrieve "question"  # hybrid retrieval, prints top 5 with citations
uv run streamlit run app.py               # demo app (run from repo root; binds 127.0.0.1:8501)
RAG_LIVE_TESTS=1 uv run pytest tests/test_live_retrieval.py -v -s  # real Gemini/Jina/Qdrant, multilingual
```

Local Qdrant: Docker container `rag-qdrant` on 127.0.0.1:6333, volume `rag_qdrant_storage`.

## Deployment (VPS)

- The app has no authentication. `.streamlit/config.toml` binds Streamlit to `127.0.0.1:8501`; keep that on the VPS and expose it only through a reverse proxy (TLS + access control such as basic auth or an IP allowlist) or an SSH tunnel. Never set `STREAMLIT_SERVER_ADDRESS=0.0.0.0` on a public interface.
- Overrides via environment: `STREAMLIT_SERVER_ADDRESS`, `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` (MB, default 200). Oversized files are rejected with a clear message.
- Keep Qdrant bound to localhost as well (`-p 127.0.0.1:6333:6333`).
- Single app process assumed. API clients are shared across Streamlit session threads on the assumption they are thread-safe (normal usage for httpx, google-genai, qdrant-client); load-test during deployment.

## Data flow notes

- Two separate collection models: the CLI (`rag.ingest`, `rag.retrieve`) uses the shared `documents` collection; the Streamlit app uses one private collection per browser session and never reads `documents`. CLI-ingested documents do not appear in the app.
- App sessions (`rag.session`): each browser session ingests into and retrieves from its own collection `session_<created>_<uuid>` (name held only in server-side session state). Last activity is stored in the collection's Qdrant metadata and refreshed by uploads, questions and app use (at most once a minute); collections inactive for 24h are deleted when a new session starts. A browser refresh starts a new, empty Streamlit session; the abandoned collection is removed by that inactivity cleanup. No accounts or persistence beyond this.
- Within a session, duplicate file names get a " (2)" suffix so `[document, Page N]` citations stay unambiguous.
- Multimodal understanding (`rag.understand`): after fast extraction, only pages flagged `scanned` (little text + large image), `garbled` (U+FFFD in text), `figure` (large figure whose caption is not an event-photo caption such as "Glimpses…", "Hon'ble…", "…held at…") or `table` (mostly empty cells) are rendered at 150 DPI and sent to MiniMax-M3. M3 output is stored as extracted document content (never an answer): scanned pages become `figure` blocks (tables stay `table`), figures get their visual content, garbled pages are re-transcribed as `text`. Any failure keeps the fast extraction. Successful results are cached in `data/cache/understanding.sqlite` (bump `PROMPT_VERSION` to invalidate).
- Embedding cache: `data/cache/embeddings.sqlite`, keyed by sha256(model, dim, task, text). Cached texts are never re-embedded.
- IDs: `document_id` = sha256(PDF bytes)[:16]; `chunk_id` = `{document_id}-p{page}-{n}`; Qdrant point id = uuid5(chunk_id).
- Re-ingesting a document upserts its points, then deletes that document's stale points.
- Generation (`rag.generate.generate_answer(query, [hit.payload for hit in hits], minimax_client(...))`): retrieved text is sent only as delimited untrusted `<source>` blocks; `<think>` is stripped; `[document, Page N]` citations not matching a supplied (document, page) are removed; no context, `INSUFFICIENT_CONTEXT`, or no valid citation → abstention. Answers use M3 with thinking disabled because M3 can start the answer inside `<think>` (MiniMax-AI/MiniMax-M3#28), which stripping cannot recover; page understanding keeps thinking on (a boundary defect there only causes a safe fallback). `Answer.raw_output` keeps the unprocessed reply for evaluation.
- The rank-bm25 index is built in memory from Qdrant chunk payloads (`rag.bm25.build_bm25(rag.store.iter_payloads(client))`), so it never drifts from the dense index. Rebuild after ingestion.

## Code navigation

Use the codebase-memory MCP tools first for structural questions (project `C-Users-tarun-projects-multimodal-enterprise-rag`): `search_graph`, `trace_path`, `get_code_snippet`, `get_architecture`. Fall back to grep for literal text. If the index looks stale after large changes, run `index_repository`.
