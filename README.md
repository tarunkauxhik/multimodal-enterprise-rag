# Multimodal Enterprise RAG

A document question-answering prototype for enterprise PDFs: upload reports in English or Hindi, ask questions in English, Hindi or Hinglish, and get answers grounded only in those documents with `[document.pdf, Page N]` citations. Pages that plain text extraction cannot handle (scans, charts, maps, broken tables) are selectively read by a multimodal model.

## Architecture

```
PDF
 → PyMuPDF4LLM fast extraction (layout-aware blocks per page)
 → selective MiniMax-M3 page understanding (only scanned / garbled / informational-figure / sparse-table pages)
 → structure-aware chunking (never across pages; section path, tables and figures kept)
 → Gemini Embedding 2 (768 dims, cached)
 → Qdrant (dense, cosine)  +  rank-bm25 (in memory, built from the same Qdrant chunks)
 → Reciprocal Rank Fusion (k=60): dense top 20 + BM25 top 20 → top 10
 → Jina reranker v3 → top 5
 → MiniMax-M3 grounded answer with validated citations
```

| Module | Role |
|---|---|
| `rag/extract.py` | PDF → pages of typed blocks (heading, text, table, figure, caption) |
| `rag/understand.py` | Page routing and MiniMax-M3 visual extraction, merged back into blocks |
| `rag/chunk.py` | Structure-aware chunks with document, page, section and content-type metadata |
| `rag/embed.py` | Gemini embeddings with a persistent SQLite cache and retries |
| `rag/store.py` | Qdrant collection, idempotent upserts |
| `rag/bm25.py` | Unicode-aware tokenizer (keeps Devanagari words whole) and BM25 search |
| `rag/retrieve.py` | Dense + BM25 → RRF → Jina rerank |
| `rag/rerank.py` | Jina reranker client |
| `rag/generate.py` | Grounded prompt, citation validation, abstention, MiniMax client |
| `rag/ingest.py` | Ingestion pipeline and CLI |
| `rag/session.py` | Per-browser-session collections, activity tracking, cleanup |
| `app.py` | Streamlit UI |

## Stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) · Streamlit · Qdrant · PyMuPDF4LLM · MiniMax-M3 (OpenAI-compatible gateway) · Gemini Embedding 2 · Jina Reranker v3 · rank-bm25 · httpx · pytest. No LangChain, LangGraph, FastAPI or Redis.

## Local setup

Prerequisites: Python 3.12 via `uv`, and Docker for Qdrant.

```bash
uv sync

# Qdrant, bound to localhost, with a persistent volume
docker run -d --name rag-qdrant -p 127.0.0.1:6333:6333 -v rag_qdrant_storage:/qdrant/storage qdrant/qdrant:latest

cp .env.example .env   # then fill in the keys

uv run streamlit run app.py    # run from the repo root; opens http://127.0.0.1:8501
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `MINIMAX_API_KEY` | yes | MiniMax-M3 page understanding and answers |
| `MINIMAX_BASE_URL` | no (defaults to the project gateway) | OpenAI-compatible MiniMax endpoint |
| `GEMINI_API_KEY` | yes | Gemini Embedding 2 |
| `JINA_API_KEY` | yes | Jina reranker |
| `QDRANT_URL` | no (default `http://127.0.0.1:6333`) | Qdrant endpoint |
| `QDRANT_API_KEY` | no | Only if Qdrant requires a key |
| `STREAMLIT_SERVER_ADDRESS` / `_PORT` / `_MAX_UPLOAD_SIZE` | no | Override `.streamlit/config.toml` (default `127.0.0.1:8501`, 200 MB) |

Keys are read only from the environment (or a local `.env`, which is gitignored) and are redacted from any error shown in the UI.

### Other commands

```bash
uv run pytest                               # offline test suite (fake services, in-memory Qdrant)
uv run python -m rag.ingest file.pdf ...    # CLI ingestion into the shared "documents" collection
uv run python -m rag.retrieve "question"    # CLI retrieval over the shared collection
RAG_LIVE_TESTS=1 uv run pytest tests/test_live_retrieval.py -v -s   # live multilingual retrieval test
```

## Major design decisions

- **Selective multimodal understanding.** Fast extraction handles every page. Only pages with strong evidence go to MiniMax-M3: little text plus a page-sized image (scan), U+FFFD-garbled text, a large figure whose caption is not an event-photo caption, or a mostly empty table. M3 output is stored as document content (figure/table blocks), never used as an answer. Failures fall back to the fast extraction, and results are cached.
- **Hybrid retrieval with one source of truth.** BM25 is rebuilt in memory from the chunks stored in Qdrant, so the sparse and dense indexes cannot drift. RRF fuses by rank, not score.
- **Grounded generation.** Retrieved text is sent as delimited untrusted data, with source tags neutralised and rules restated after it, so instructions inside documents are ignored. Every `[document, Page N]` citation is checked against the retrieved chunks; unknown citations are removed and reported, and an answer with no valid citation becomes an abstention (`INSUFFICIENT_CONTEXT`).
- **M3 thinking disabled for answers.** With thinking on, M3 can start its answer inside `<think>` ([MiniMax-M3#28](https://github.com/MiniMax-AI/MiniMax-M3/issues/28)), so stripping the reasoning would drop the opening words. `<think>` stripping stays as a safety net; raw output is kept for debugging but never shown.
- **Session isolation without accounts.** Each browser session ingests into and searches its own Qdrant collection, named on the server. Last activity is stored in collection metadata; collections idle for 24 hours are deleted. The CLI uses a separate shared collection that the app never reads.
- **Idempotent, cached ingestion.** Document IDs are content hashes, chunk and point IDs are derived from them, and embeddings and M3 page results are cached, so re-ingesting a PDF does not re-embed or re-call M3.
- **Localhost by default.** The app has no authentication, so Streamlit binds to `127.0.0.1`.

## Evaluation data

`evals/gold.jsonl` is a 30-question gold set (English, Hindi, Hinglish, cross-language, tables, visual pages, unanswerable and prompt-injection cases) built from three public annual reports. See [evals/README.md](evals/README.md). The source PDFs are not committed, and an evaluation runner is not part of this prototype yet.

## Known limitations

- **No authentication.** Session isolation is not access control. On a server, keep Streamlit on localhost behind a reverse proxy with TLS and access control, or use an SSH tunnel.
- **Legacy Hindi fonts are unreadable.** PDFs set in non-Unicode Hindi fonts (e.g. Arjun, BHARTIYA-HINDI_081) extract as Latin gibberish and are not detected as garbled, so they are neither re-read by M3 nor usefully searchable.
- **Unicode Hindi extraction is lossy.** PyMuPDF drops parts of some conjunct characters and doubles some vowel signs, which weakens Hindi keyword (BM25) matching; numbers usually survive.
- **Page routing is heuristic.** Uncaptioned photos and decorative full-page images are still sent to M3; vector-drawn charts without an embedded image are not detected. Thresholds are not yet tuned on real corpora.
- **Hinglish reranking.** Jina reranker v3 can demote a correct romanised-Hindi → Devanagari match (one known case in the live test is marked `xfail`).
- **Refreshing the page starts a new, empty session.** The old documents are removed by the 24-hour inactivity cleanup.
- **Single process, not load-tested.** API clients are shared across Streamlit sessions on the assumption that they are thread-safe; ingestion runs inside the uploading user's session.
- **Not yet evaluated end to end.** The gold set exists, but retrieval and answer quality have not been measured on it.
