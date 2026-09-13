# Multimodal Enterprise RAG

Question answering over enterprise PDFs. Upload English or Hindi reports, ask in English, Hindi or Hinglish, and get answers grounded only in those documents, with validated `[document.pdf, Page N]` citations.

**Live:** [tarun.runs-on.dev](https://tarun.runs-on.dev)  
**Stack:** Python 3.12 · Streamlit · Qdrant · PyMuPDF4LLM · MiniMax-M3 · Gemini Embedding 2 · rank-bm25 · Jina Reranker v3

## How it works

```mermaid
flowchart LR
    subgraph Ingest
        P[PDF] --> X[PyMuPDF4LLM<br/>fast extraction]
        X --> R{Hard page?}
        R -- no --> C[Structure-aware<br/>chunking]
        R -- yes --> M[MiniMax-M3<br/>page vision] --> C
        C --> E[Gemini Embedding 2<br/>768d]
        E --> Q[(Qdrant)]
    end

    subgraph Query
        U[Question] --> D[Dense top 20]
        U --> B[BM25 top 20]
        D --> F[RRF k=60<br/>top 10]
        B --> F
        F --> J[Jina rerank<br/>top 5]
        J --> G[MiniMax-M3<br/>grounded answer]
        G --> V[Citation check]
        V --> A[Answer + citations<br/>or abstain]
    end

    Q --> D
    Q -. chunk payloads .-> B
```

1. **Extract** every page quickly with PyMuPDF4LLM into typed blocks (heading, text, table, figure, caption).
2. **Route** only hard pages to MiniMax-M3 vision; everything else keeps the fast extraction.
3. **Chunk** by structure: chunks never cross pages, keep their section path, and tables and figures stay whole.
4. **Embed** with Gemini Embedding 2 (768d, cached in SQLite) and store in Qdrant (cosine).
5. **Retrieve** with dense search plus BM25, fuse by rank with RRF, rerank with Jina.
6. **Answer** with MiniMax-M3 from the top 5 chunks, then validate every citation.

## Selective multimodal understanding

Sending every page to a vision model is slow: a 9-page M3 batch took about 30 s. So a page goes to M3 only when a deterministic check says fast extraction may have lost something:

| Signal | Rule | M3 output stored as |
|---|---|---|
| Scanned | < 50 non-space characters and an image covering ~40% of the page | `figure` blocks |
| Garbled | ≥ 1% U+FFFD replacement characters | re-transcribed `text` |
| Informational figure | large figure whose caption is not an event photo ("Glimpses…", "Hon'ble…", "…held at…") | figure content |
| Broken table | ≥ 50% of data cells empty | `table` |

Pages are rendered at 150 DPI. M3 output is stored as **document content, never as an answer**, then chunked and embedded like any other block. If M3 fails, the fast extraction is kept. Results are cached in `data/cache/understanding.sqlite`.

## Retrieval and grounding

- **Hybrid search.** Dense retrieval handles paraphrase and cross-language queries; BM25 catches exact names, numbers and terms. Its tokenizer is Unicode-aware, so Devanagari words stay whole.
- **One source of truth.** The BM25 index is rebuilt in memory from the chunks stored in Qdrant, so the two indexes cannot drift.
- **RRF over ranks.** Dense and BM25 scores are not comparable, so they are fused by rank (k=60).
- **Untrusted context.** Retrieved text is sent inside delimited `<source>` blocks, with the rules restated after it, so instructions hidden in a PDF are ignored.
- **Citation validation.** Each `[document, Page N]` must match a retrieved chunk. A bare `[Page N]` is accepted only if exactly one document has that page. Invalid citations are removed; if none remain, the system answers `INSUFFICIENT_CONTEXT`.
- **Thinking disabled for answers.** M3 can start its answer inside `<think>` ([MiniMax-M3#28](https://github.com/MiniMax-AI/MiniMax-M3/issues/28)), which would cut the opening words, so answers run with thinking off.

## Design choices

| Component | Chosen | Tried | Why |
|---|---|---|---|
| PDF extraction | PyMuPDF4LLM (~6.6 s) | Docling (56–185 s) | Docling was accurate but too slow for a prototype |
| Vision | MiniMax-M3 | MiniMax-M2.7 | M2.7 failed the tested charts and images |
| Embeddings | Gemini Embedding 2 (19 chunks ~1.7 s) | BGE-M3, BGE-small/base, Qwen3-Embedding-0.6B | Local models too slow or English-focused |
| Reranker | Jina Reranker v3 (19 pairs ~1.45 s) | BGE reranker v2-m3, Voyage | Too slow locally / rate limits |
| Orchestration | Plain Python modules | — | No LangChain, LangGraph or FastAPI; the pipeline is linear and easy to debug |

Other decisions:

- **Idempotent ingestion.** `document_id` is the first 16 hex characters of the PDF's SHA-256, and chunk and point IDs derive from it. Re-ingesting upserts, then deletes stale chunks. Cached embeddings are never recomputed, so an interrupted ingestion resumes cheaply.
- **Session isolation.** Each browser session gets its own Qdrant collection, deleted after 24 h of inactivity. The CLI uses a separate shared `documents` collection that the app never reads.

## Project structure

```text
app.py              Streamlit UI
rag/
  config.py         models, constants, settings from env
  extract.py        PDF -> typed page blocks
  understand.py     page routing + MiniMax-M3 vision
  chunk.py          structure-aware chunks with metadata
  embed.py          Gemini embeddings + SQLite cache
  store.py          Qdrant collection and upserts
  bm25.py           Unicode-aware BM25
  retrieve.py       dense + BM25 -> RRF -> rerank
  rerank.py         Jina client
  generate.py       grounded prompt, citation validation, M3 client
  ingest.py         ingestion pipeline + CLI
  session.py        per-session collections and cleanup
tests/              offline pytest suite
evals/              30-question gold set
```

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
docker run -d --name rag-qdrant -p 127.0.0.1:6333:6333 \
  -v rag_qdrant_storage:/qdrant/storage qdrant/qdrant:latest
cp .env.example .env          # fill in the API keys
uv run streamlit run app.py   # http://127.0.0.1:8501
```

Qdrant must be running before the app starts.

| Variable | Required | Purpose |
|---|---|---|
| `MINIMAX_API_KEY` | yes | Page understanding and answers |
| `GEMINI_API_KEY` | yes | Embeddings |
| `JINA_API_KEY` | yes | Reranking |
| `MINIMAX_BASE_URL` | no | OpenAI-compatible MiniMax endpoint |
| `QDRANT_URL` | no | Default `http://127.0.0.1:6333` |
| `QDRANT_API_KEY` | no | Only if Qdrant requires one |

Keys are read only from the environment or a gitignored `.env`, and are redacted from errors shown in the UI.

```bash
uv run pytest                               # offline suite: 143 passed, 7 live tests skipped
uv run python -m rag.ingest file.pdf ...    # CLI ingestion (shared collection)
uv run python -m rag.retrieve "question"    # CLI retrieval
RAG_LIVE_TESTS=1 uv run pytest tests/test_live_retrieval.py -v -s
```

The live test sends English, Hindi, Hinglish and cross-language queries through real Gemini, Qdrant and Jina. One Hinglish → Devanagari case is marked `xfail` because Jina demotes the correct chunk out of first place.

## Deployment

```text
Internet -> Nginx (HTTPS, Let's Encrypt) -> Streamlit 127.0.0.1:8501 -> Qdrant 127.0.0.1:6333
                                                    |
                                                    +-> MiniMax, Gemini, Jina APIs
```

Runs on an OCI VM with Docker and systemd. Streamlit and Qdrant both bind to localhost; only Nginx is public. The app has no authentication, so access control belongs at the proxy.

## Evaluation

[`evals/gold.jsonl`](evals/gold.jsonl) holds 30 questions over three public annual reports: English, Hindi, Hinglish, cross-language, tables, visual pages, unanswerable questions and prompt injection. The PDFs are not committed. No end-to-end score (accuracy, Recall@K, faithfulness) is claimed yet; the set exists so those can be measured reproducibly.

## Limitations

- No authentication; session isolation is not access control.
- Legacy non-Unicode Hindi fonts (e.g. Arjun, BHARTIYA-HINDI_081) extract as Latin gibberish and are not flagged as garbled.
- Unicode Hindi extraction drops parts of some conjuncts and doubles some vowel signs, which weakens BM25.
- Routing thresholds are heuristic: uncaptioned or decorative images can still reach M3, and vector-drawn charts are missed.
- Refreshing the page starts a new, empty session.
- Single process, not load-tested.
