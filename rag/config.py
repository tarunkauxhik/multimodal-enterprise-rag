"""Runtime settings.

Benchmarked model/retrieval choices are constants: changing them is an
architecture decision, not a deployment knob. Secrets and per-environment
endpoints come only from environment variables (optionally a local .env).
Secret values are never included in repr, logs, or error messages.
"""

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"  # gitignored runtime data
EMBED_CACHE_PATH = DATA_DIR / "cache" / "embeddings.sqlite"
UNDERSTAND_CACHE_PATH = DATA_DIR / "cache" / "understanding.sqlite"

# --- Benchmarked decisions (see CLAUDE.md before changing) ---
# LLM for page understanding, follow-up rewriting and answers: MiniMax-M3 over the OpenAI-compatible
# Chat Completions API (POST {MINIMAX_BASE_URL}/chat/completions). MINIMAX_BASE_URL is required and
# has no default, so a missing setting fails at startup instead of calling an unintended endpoint.
# Chosen over MiniMax-M3.1-Flash in an A/B on this pipeline's own tasks: equal grounded answers and
# rewrites, but M3 transcribed real hard pages (legacy Hindi font, a scan) more completely and
# accurately, at lower latency.
LLM_PROVIDER = "minimax"
LLM_MODEL = "MiniMax-M3"
# Thinking disabled for every call (thinking={"type": "disabled"}, supported by MiniMax-M3). With
# thinking on, M3 returns its reasoning inside `content` as <think>...</think> (seen in testing)
# and can start the answer before closing it (github.com/MiniMax-AI/MiniMax-M3/issues/28), which
# stripping cannot recover. Grounded extraction and transcription do not need it.
LLM_THINKING = False
# max_completion_tokens: MiniMax counts any thinking tokens against it; with thinking off it is the answer.
GENERATION_MAX_TOKENS = 4096  # not benchmarked
UNDERSTAND_MAX_TOKENS = 8192  # page transcription can be long; not benchmarked
UNDERSTAND_DPI = 150  # page render resolution sent to the model (PNG; MiniMax accepts images up to 10 MB)
# Follow-up rewriting (rag.contextualize): one short call, only for messages that have earlier turns.
# Measured with MiniMax-M3 (thinking off): median ~1.5 s per call, with rare spikes past 20 s, hence
# one attempt and a short timeout; on any failure the original message is used.
HISTORY_TURNS = 3  # earlier exchanges used to resolve a follow-up
HISTORY_ANSWER_CHARS = 1500  # each earlier answer is cut to this length in the rewrite prompt
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

REQUIRED_ENV = ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "GEMINI_API_KEY", "JINA_API_KEY")


@dataclass(frozen=True)
class Settings:
    minimax_api_key: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    jina_api_key: str = field(repr=False)
    minimax_base_url: str
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


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_base_url(url: str) -> None:
    """The API key travels in a header, so a remote endpoint must be HTTPS; HTTP is for local development only."""
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise RuntimeError("MINIMAX_BASE_URL must be an https URL ending in the API version, e.g. .../v1")
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise RuntimeError("MINIMAX_BASE_URL must use https; http is allowed only for localhost or a loopback address")


def load_settings(env_file: Path | None = ROOT / ".env") -> Settings:
    """Build Settings from the environment. Raises naming missing keys only."""
    if env_file is not None:
        _load_env_file(env_file)

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    base_url = os.environ["MINIMAX_BASE_URL"].strip().rstrip("/")
    _check_base_url(base_url)

    return Settings(
        minimax_api_key=os.environ["MINIMAX_API_KEY"].strip(),
        gemini_api_key=os.environ["GEMINI_API_KEY"].strip(),
        jina_api_key=os.environ["JINA_API_KEY"].strip(),
        minimax_base_url=base_url,
        qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY", "").strip() or None,
    )
