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

# --- Benchmarked decisions (see CLAUDE.md before changing) ---
MINIMAX_MODEL = "MiniMax-M3"

GEMINI_EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
EMBED_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"
EMBED_BATCH_SIZE = 5  # sized for our Gemini API quota; 429s are retried with backoff

JINA_RERANK_MODEL = "jina-reranker-v3"

QDRANT_COLLECTION = "documents"  # cosine distance, EMBED_DIM vectors

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RRF_TOP_K = 10  # fused candidates sent to Jina
RERANK_TOP_K = 5  # chunks sent to MiniMax for the answer

REQUIRED_SECRETS = ("MINIMAX_API_KEY", "GEMINI_API_KEY", "JINA_API_KEY")


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


def load_settings(env_file: Path | None = ROOT / ".env") -> Settings:
    """Build Settings from the environment. Raises naming missing keys only."""
    if env_file is not None:
        _load_env_file(env_file)

    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        minimax_api_key=os.environ["MINIMAX_API_KEY"].strip(),
        gemini_api_key=os.environ["GEMINI_API_KEY"].strip(),
        jina_api_key=os.environ["JINA_API_KEY"].strip(),
        minimax_base_url=os.environ.get(
            "MINIMAX_BASE_URL", "https://llm-gateway-azure.penpencil.guru/v1"
        ).rstrip("/"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY", "").strip() or None,
    )
