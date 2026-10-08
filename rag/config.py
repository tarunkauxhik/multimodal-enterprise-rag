"""Runtime settings.

Benchmarked model/retrieval choices are constants: changing them is an
architecture decision, not a deployment knob. Secrets and per-environment
endpoints come only from environment variables (optionally a local .env).
Secret values are never included in repr, logs, or error messages.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"  # gitignored runtime data
EMBED_CACHE_PATH = DATA_DIR / "cache" / "embeddings.sqlite"
UNDERSTAND_CACHE_PATH = DATA_DIR / "cache" / "understanding.sqlite"

# --- Benchmarked decisions (see CLAUDE.md before changing) ---
# LLM for page understanding and answers: xAI Grok 4.7 over its OpenAI-compatible Chat Completions
# API (POST {XAI_BASE_URL}/chat/completions). Temporary replacement for MiniMax-M3, whose gateway is
# unavailable; the pipeline was benchmarked with MiniMax-M3, not with Grok.
LLM_PROVIDER = "xai"
LLM_MODEL = "grok-4.7"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
# Grok 4.7 always reasons ("Reasoning cannot be disabled"; efforts low|medium|high (default)|xhigh,
# docs.x.ai). Reasoning is returned separately (message.reasoning_content), never inside the answer
# text. Answers are grounded extraction from five supplied chunks and page understanding is
# transcription: neither needs deep multi-step reasoning, so both use "low" for latency and cost.
GENERATION_REASONING_EFFORT = "low"
UNDERSTAND_REASONING_EFFORT = "low"
# max_completion_tokens: caps visible output only; reasoning tokens are not counted against it.
GENERATION_MAX_TOKENS = 4096  # not benchmarked
UNDERSTAND_MAX_TOKENS = 8192  # page transcription can be long; not benchmarked
UNDERSTAND_DPI = 150  # page render resolution sent to the vision model (PNG; xAI accepts PNG/JPEG up to 20 MiB)
# Follow-up rewriting (rag.contextualize): one short call, only for messages that have earlier turns.
# Measured with grok-4.7 at low effort: median ~2 s per rewrite, with rare spikes of 20-40 s, hence one
# attempt and a short timeout; on any failure the original message is used.
HISTORY_TURNS = 3  # earlier exchanges used to resolve a follow-up
HISTORY_ANSWER_CHARS = 1500  # each earlier answer is cut to this length in the rewrite prompt
REWRITE_REASONING_EFFORT = "low"
REWRITE_MAX_TOKENS = 300  # one standalone request
REWRITE_TIMEOUT = 15.0  # seconds

GEMINI_EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
EMBED_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"
EMBED_BATCH_SIZE = 5  # sized for our Gemini API quota; 429s are retried with backoff

JINA_RERANK_MODEL = "jina-reranker-v3"

# Shared collection for CLI ingestion/retrieval (rag.ingest, rag.retrieve) and, by default, the
# HTTP API's workspace (API_COLLECTION), so CLI-ingested documents appear there.
# Cosine distance, EMBED_DIM vectors.
QDRANT_COLLECTION = "documents"

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RRF_TOP_K = 10  # fused candidates sent to Jina
RERANK_TOP_K = 5  # chunks sent to the LLM for the answer

REQUIRED_SECRETS = ("XAI_API_KEY", "GEMINI_API_KEY", "JINA_API_KEY")


@dataclass(frozen=True)
class Settings:
    xai_api_key: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    jina_api_key: str = field(repr=False)
    xai_base_url: str
    qdrant_url: str
    qdrant_api_key: str | None = field(default=None, repr=False)


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding real env vars."""
    # ponytail: minimal .env reader (no multiline/interpolation); switch to python-dotenv if needed
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_settings(env_file: Path | None = ROOT / ".env") -> Settings:
    """Build Settings from the environment. Raises naming missing keys only."""
    if env_file is not None:
        _load_env_file(env_file)

    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        xai_api_key=os.environ["XAI_API_KEY"].strip(),
        gemini_api_key=os.environ["GEMINI_API_KEY"].strip(),
        jina_api_key=os.environ["JINA_API_KEY"].strip(),
        xai_base_url=(os.environ.get("XAI_BASE_URL", "").strip() or DEFAULT_XAI_BASE_URL).rstrip("/"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY", "").strip() or None,
    )
