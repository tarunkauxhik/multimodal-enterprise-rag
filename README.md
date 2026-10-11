# Multimodal Enterprise RAG

A deployed question-answering app for PDFs. Upload documents, ask questions in English, Hindi or Hinglish, and get answers drawn only from those documents, with page citations and the passages behind them. Follow-up questions use the recent conversation.

Live demo: [tarun.runs-on.dev](https://tarun.runs-on.dev) · Code: [github.com/tarunkauxhik/multimodal-enterprise-rag](https://github.com/tarunkauxhik/multimodal-enterprise-rag)

## What it does

- Extracts every page fast, and sends only hard pages (scans, broken fonts, charts, sparse tables) to a vision model.
- Retrieves with dense search and BM25 together, fuses the rankings with RRF, and reranks with Jina.
- Cites `[document, Page N]` and checks every citation against the passages the model saw.
- Resolves follow-ups like "tell me more" or "what about page 4?" from the last 3 exchanges.
- Abstains when the documents don't cover a question, and refuses general-knowledge questions.

## Architecture

### High level design

```mermaid
flowchart LR
    U["Browser"] --> N
    subgraph VM["Oracle Cloud VM"]
        N["Nginx<br/>:443"] --> W["Next.js<br/>:3000"] --> A["FastAPI<br/>:8000"]
        A --> Q[("Qdrant<br/>:6333")]
        A --> C[("SQLite<br/>caches")]
    end
    A -.-> X["External APIs<br/>Grok · Gemini · Jina"]
```

Next.js renders the UI and forwards five allowlisted endpoints to FastAPI, where all RAG logic lives. Grok reads hard pages, rewrites follow-ups and writes answers; Gemini embeds; Jina reranks.

### Low level design

```mermaid
flowchart TB
    subgraph ING["Ingestion: ingest"]
        direction LR
        EX["extract<br/>PyMuPDF4LLM"] --> UN["understand<br/>Grok, hard pages"] --> CH["chunk<br/>page-bound"] --> EM["embed<br/>Gemini 768d"] --> ST[("store<br/>Qdrant")]
    end

    subgraph ASK["Question answering: session.respond"]
        direction TB
        MSG["Message +<br/>last 3 exchanges"] --> CX["contextualize<br/>Grok rewrite, follow-ups only"]
        CX --> RT{"route"}
        RT -->|"small talk,<br/>metadata"| DIRECT["Direct reply,<br/>no retrieval"]
        RT -->|"page, overview"| PICK["Chosen chunks"]
        RT -->|"question"| DN
        RT -->|"question"| BM
        subgraph HYB["Hybrid retrieval: retrieve, scoped to a named document"]
            direction LR
            DN["Dense, Qdrant<br/>semantic similarity<br/>top 20"]
            BM["BM25, in memory<br/>keyword matching<br/>top 20"]
        end
        DN --> RRF["RRF k=60<br/>top 10"]
        BM --> RRF
        RRF --> JR["Jina rerank<br/>top 5"]
        JR --> GEN["generate, Grok<br/>typed question + rewrite"]
        PICK --> GEN
        GEN --> VAL{"valid<br/>citations?"}
        VAL -->|"yes"| ANS["Answer +<br/>sources"]
        VAL -->|"no"| ABS["Abstain"]
    end

    ING ~~~ ASK
```

Each box is a plain Python module in `rag/`, with no LangChain or agent framework.

### Workflows

```mermaid
flowchart TB
    subgraph Q["Ask a question"]
        direction TB
        Q1["Browser tab<br/>history in sessionStorage"] --> Q2["POST /api/chat<br/>question + last 3 exchanges"]
        Q2 --> Q3["Next.js proxy<br/>forwards to FastAPI"]
        Q3 --> Q4["Reply kind: answer, abstain,<br/>out of scope, clarify,<br/>workspace or conversation"]
        Q4 --> Q5["Proxy strips internals,<br/>UI numbers citations"]
    end
    subgraph UP["Upload a PDF"]
        direction TB
        U1["Documents page<br/>upload"] --> U2{"Same PDF already<br/>stored and complete?"}
        U2 -->|"yes"| U3["Ready,<br/>nothing re-processed"]
        U2 -->|"no"| U4["Queued<br/>one runs, one waits,<br/>more get 429"]
        U4 --> U5["Background worker<br/>runs the ingestion stages"]
        U5 --> U6["Status: ready, incomplete,<br/>empty or failed<br/>UI polls every 2 s"]
    end
```

The workflows show the same two paths from the user's side: upload statuses, the chat request and the reply kinds.

## How it works

### Document ingestion

- **PDF extraction.** PyMuPDF4LLM turns every page into typed blocks: headings, text, tables, figures and captions.
- **Selective multimodal processing.** Grok reads a page image only when the fast text is likely incomplete:
  - a scan (under 50 characters and a large image);
  - garbled text (≥ 1% U+FFFD);
  - mostly a legacy Hindi font such as Kruti Dev;
  - a large figure that isn't an event photo;
  - a table with half its cells empty.

  Its output becomes document content, never an answer. A failed call keeps the fast text.
- **Chunking, embeddings and storage.** A chunk never spans pages and keeps its section path. Tables and figures stay whole, and prose is packed up to 1,500 characters. Chunks are embedded with Gemini and upserted to Qdrant (cosine), and re-ingesting deletes the document's stale points.

### Question answering

- **Conversation contextualization.** With earlier turns, one short Grok call rewrites the message into a standalone request ("and the next page?" becomes "What is on page 4 of SRS.pdf?"). First messages and small talk skip it, and a new topic comes back unchanged.
- **Routing and document scoping.** A deterministic router answers small talk and metadata questions directly. Page and overview questions get their chunks without search, since "page 72" has nothing to match. A question that names a document searches only that document.
- **Hybrid retrieval.** Dense search covers paraphrase and cross-language questions. BM25 catches exact names and numbers, and its tokenizer keeps Devanagari words whole. BM25 is rebuilt from Qdrant payloads, so the two indexes can't drift.
- **RRF and reranking.** The scores aren't comparable, so ranks are fused with RRF. Jina then keeps the best 5 for the answer.
- **Grounded generation.** Sources go into delimited, untrusted `<source>` blocks. The generator answers the message as typed, with the rewrite attached, so the user's own instructions survive. Earlier answers never become evidence.
- **Citation validation.** Each `[document, Page N]` must match a supplied passage, and invalid citations are removed. With none left, the app abstains. `INSUFFICIENT_CONTEXT` (not enough evidence) and `OUT_OF_SCOPE` (not about the documents) are separate refusals, and neither falls back to general knowledge.

### Caching and reliability

- **`embeddings.sqlite`** is keyed by sha256(model, dimensions, task, text) and stores the vector. Chunks and questions are cached alike, so repeated text is never re-embedded.
- **`understanding.sqlite`** is keyed by sha256(provider/model, prompt version, DPI, document, page, task) and stores Grok's JSON result. Only successes are cached.
- **Reuse.** Re-ingesting a PDF reuses both caches, so even an interrupted run skips finished work. A new model or prompt version changes the key and forces a recompute. Neither cache expires or has a cleanup command.
- **Write completeness.** Every Qdrant write is tagged with a `write_id` and chunk count, so a write cut short by a crash is never retrieved.
- **Failure handling.**
  - Grok and Jina calls retry 408, 429, 5xx and network errors with backoff, and Gemini retries through its SDK.
  - A reply cut off at the token limit raises an error instead of returning half an answer.
  - A failed rewrite (one attempt, 15 s timeout) keeps the typed message.
  - Questions are capped at 2,000 characters and history at 3 turns.
  - All Grok calls run at low reasoning effort.

## Sessions, documents and deletion

- **Chat history** lives in the browser tab's sessionStorage. It survives a reload and is gone when the tab closes. The server stores no conversations; each request carries its last 3 exchanges, with answers cut to 1,500 characters. Separate tabs don't share history. This is not user identity or access control.
- **Documents** live in one shared Qdrant collection (`documents`). There are no accounts, so anyone who can reach the site can list, query and delete every document. A duplicate file name gets a " (2)" suffix so citations stay unambiguous.
- **Deletion** removes all of the document's Qdrant points, and with them its listing and retrieval. The uploaded PDF is never written to disk. The SQLite caches are not touched, so the document's cached page results (extracted text) and embedding vectors stay on disk.

## Deployment

Only Nginx is public: port 443 with a certbot certificate, and port 80 redirects to HTTPS. Next.js (`127.0.0.1:3000`), FastAPI (`127.0.0.1:8000`, one Uvicorn worker because upload jobs live in memory) and Qdrant in Docker (port 6333) are not reachable from the internet. The `rag-api` and `rag-web` systemd units and the Nginx site config are in [`deploy/`](deploy/).

GitHub Actions CI runs on every pull request and push to `main`. The backend job runs `uv sync --frozen` and pytest. The frontend job runs `npm ci`, Vitest, ESLint, `tsc` and a production build.

CD runs only after CI passes on `main`. Over SSH it refuses a dirty working tree, checks out the exact commit CI tested, and runs `uv sync --frozen`, `npm ci` and `npm run build`. It then restarts both services and waits up to 30 s each for the API and web app to answer locally.

**Access control.** The app implements no authentication or authorization. [`deploy/nginx.conf`](deploy/nginx.conf) enables Nginx basic auth for the whole site, but the live demo currently loads without a login. Treat anything uploaded there as public.

## Tech stack

| Area | Used |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui |
| Backend | Python 3.12, FastAPI, httpx, uv |
| Models | xAI `grok-4.7`, Google `gemini-embedding-2` (768 dims), Jina `jina-reranker-v3` |
| Documents and retrieval | PyMuPDF4LLM, Qdrant (cosine), rank-bm25, SQLite caches |
| Ops | Oracle Cloud VM, Nginx, systemd, Docker, GitHub Actions |

## Run locally

You need [uv](https://docs.astral.sh/uv/), Node.js 22.22+ or 24.15+, Docker, and keys for xAI, Gemini and Jina. The API won't start without the keys; the offline tests don't need them.

```bash
uv sync
docker run -d --name rag-qdrant -p 127.0.0.1:6333:6333 -v rag_qdrant_storage:/qdrant/storage qdrant/qdrant:latest
cp .env.example .env        # Windows: copy .env.example .env, then add your keys
uv run uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Then, in a second terminal:

```bash
cd web
npm ci
npm run dev -- --hostname 127.0.0.1    # http://127.0.0.1:3000
```

`.env.example` lists the variables. The required ones are `XAI_API_KEY`, `GEMINI_API_KEY` and `JINA_API_KEY`. The optional ones (`XAI_BASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`) default to the public xAI endpoint and local Qdrant.

## Testing

```bash
uv run pytest                                   # backend, offline
cd web && npm test && npm run lint && npm run typecheck && npm run build
RAG_LIVE_TESTS=1 uv run pytest tests/test_live_retrieval.py -v -s   # opt-in, real APIs
```

The backend suite runs against in-memory Qdrant with fake models. It covers extraction, page routing, chunking, RRF, citation validation, the follow-up rewrite, the HTTP API and write completeness. The frontend tests use Vitest and Testing Library, with proxy validation and an axe-core accessibility check. The live multilingual retrieval test is skipped in CI.

## Evaluation

[`evals/gold.jsonl`](evals/gold.jsonl) has 30 questions over three public annual reports (NITI Aayog in English and Hindi, and a bilingual Ministry of Heavy Industries report). They cover English, Hindi, Hinglish, cross-language, tables, visual pages, one unanswerable question and one prompt injection.

`uv run python -m evals.benchmark` compares dense, BM25, RRF and RRF + Jina on retrieval only. It reports Recall@5, Recall@10, MRR@10 and nDCG@10 into `evals/results/`. It needs the PDFs in `evals/docs/` (not committed), a running Qdrant and the keys in `.env`. Only Gemini and Jina are called. No results have been recorded yet, so no scores are claimed.

## Limitations

- No accounts and one shared workspace (see Access control).
- Grok 4.7 replaced the benchmarked MiniMax-M3 without a re-run.
- Citation checks confirm a cited page was supplied, not that it supports the sentence.
- Page and chat routing are heuristics. Vector-drawn charts are missed.
- Follow-ups see only 3 exchanges. Document scoping applies only when the rewrite names the file.
- Single process, in-memory job state, not load-tested.
