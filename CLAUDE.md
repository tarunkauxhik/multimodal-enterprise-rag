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
| Generation | MiniMax-M3 over the top 5 chunks |
| Citations | `[Page X]` for now; chunk metadata carries document, page, section for later `[document, page, section]` |

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
```

## Data flow notes

- Embedding cache: `data/cache/embeddings.sqlite`, keyed by sha256(model, dim, task, text). Cached texts are never re-embedded.
- IDs: `document_id` = sha256(PDF bytes)[:16]; `chunk_id` = `{document_id}-p{page}-{n}`; Qdrant point id = uuid5(chunk_id).
- Re-ingesting a document upserts its points, then deletes that document's stale points.
- The rank-bm25 index is built in memory from Qdrant chunk payloads (`rag.bm25.build_bm25(rag.store.iter_payloads(client))`), so it never drifts from the dense index. Rebuild after ingestion.

## Code navigation

Use the codebase-memory MCP tools first for structural questions (project `C-Users-tarun-projects-multimodal-enterprise-rag`): `search_graph`, `trace_path`, `get_code_snippet`, `get_architecture`. Fall back to grep for literal text. If the index looks stale after large changes, run `index_repository`.
