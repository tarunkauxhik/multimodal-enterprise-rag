import httpx
import pytest

from rag import embed as embed_module
from rag.config import EMBED_DIM
from rag.embed import EmbeddingCache, embed_texts, gemini_embedder

TASK = "RETRIEVAL_DOCUMENT"
KEY = "gemini_secret_key_123"


class FakeEmbedder:
    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t))] + [1.0] * (EMBED_DIM - 1) for t in texts]


def test_embeds_unique_texts_in_batches_and_keeps_order(tmp_path):
    fake, cache = FakeEmbedder(), EmbeddingCache(tmp_path / "e.sqlite")
    texts = ["a", "bb", "a", "ccc", "dddd", "bb"]
    vectors, new = embed_texts(texts, TASK, fake, cache, batch_size=2)
    assert new == 4
    assert [len(b) for b in fake.batches] == [2, 2]
    assert [v[0] for v in vectors] == [1.0, 2.0, 1.0, 3.0, 4.0, 2.0]
    assert all(len(v) == EMBED_DIM for v in vectors)


def test_cache_persists_and_is_never_re_embedded(tmp_path):
    path = tmp_path / "e.sqlite"
    first = EmbeddingCache(path)
    vectors1, _ = embed_texts(["x", "yy"], TASK, FakeEmbedder(), first)
    first.close()

    fake = FakeEmbedder()
    vectors2, new = embed_texts(["yy", "x"], TASK, fake, EmbeddingCache(path))
    assert new == 0 and fake.batches == []
    assert vectors2 == [vectors1[1], vectors1[0]]


def test_only_missing_texts_are_embedded(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    embed_texts(["old"], TASK, FakeEmbedder(), cache)
    fake = FakeEmbedder()
    _, new = embed_texts(["old", "new"], TASK, fake, cache)
    assert new == 1 and fake.batches == [["new"]]


def test_task_type_is_part_of_cache_key(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    embed_texts(["same"], TASK, FakeEmbedder(), cache)
    fake = FakeEmbedder()
    embed_texts(["same"], "RETRIEVAL_QUERY", fake, cache)
    assert fake.batches == [["same"]]


@pytest.mark.parametrize(
    "bad",
    [lambda texts: [], lambda texts: [[0.1] * 3 for _ in texts]],
    ids=["wrong-count", "wrong-dim"],
)
def test_bad_api_response_raises_and_caches_nothing(tmp_path, bad):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    with pytest.raises(RuntimeError, match="Embedding API returned"):
        embed_texts(["a"], TASK, bad, cache)
    fake = FakeEmbedder()
    embed_texts(["a"], TASK, fake, cache)
    assert fake.batches == [["a"]]


# --- gemini_embedder retries (no network: the SDK client is faked) -----------


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeModels:
    """Raises/returns the scripted outcomes in order, recording each call."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def embed_content(self, *, model, contents, config):
        self.calls.append(contents)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return type("Result", (), {"embeddings": [FakeEmbedding([0.5] * EMBED_DIM) for _ in contents]})()


def fake_gemini(monkeypatch, outcomes):
    """Patch genai.Client so gemini_embedder builds against the fake, and capture its kwargs."""
    models, created = FakeModels(outcomes), {}

    def client(**kwargs):
        created.update(kwargs)
        return type("Client", (), {"models": models})()

    monkeypatch.setattr(embed_module.genai, "Client", client)
    return models, created


def test_http_status_retry_options_are_still_handed_to_the_sdk(monkeypatch):
    _, created = fake_gemini(monkeypatch, [None])
    gemini_embedder(KEY, TASK)(["x"])
    retry = created["http_options"].retry_options
    assert retry.http_status_codes == [408, 429, 500, 502, 503, 504]  # 429/5xx still retried by the SDK
    assert retry.attempts == 8 and retry.jitter == 1.0
    assert created["api_key"] == KEY


def test_transient_transport_error_is_retried_and_then_succeeds(monkeypatch):
    models, _ = fake_gemini(monkeypatch, [httpx.RemoteProtocolError("Server disconnected"), None])
    sleeps = []

    vectors = gemini_embedder(KEY, TASK, sleep=sleeps.append)(["a", "b"])

    assert [len(v) for v in vectors] == [EMBED_DIM, EMBED_DIM]
    assert len(models.calls) == 2 and len(sleeps) == 1  # one failure, one backoff, one success
    assert [len(c) for c in models.calls] == [2, 2]  # one Content per text, unchanged


def test_ssl_read_error_is_retried(monkeypatch):
    """The failure that killed the benchmark ingest: a TLS-level ReadError, no HTTP status."""
    models, _ = fake_gemini(monkeypatch, [httpx.ReadError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC]"), None])
    assert len(gemini_embedder(KEY, TASK, sleep=lambda s: None)(["a"])) == 1
    assert len(models.calls) == 2


def test_repeated_transport_failures_raise_after_the_attempt_limit(monkeypatch):
    models, _ = fake_gemini(monkeypatch, [httpx.ConnectError(f"boom {KEY}")] * 5)
    sleeps = []

    with pytest.raises(RuntimeError) as exc:
        gemini_embedder(KEY, TASK, attempts=5, sleep=sleeps.append)(["a"])

    assert "ConnectError" in str(exc.value) and KEY not in str(exc.value)  # key never leaks into the error
    assert len(models.calls) == 5 and len(sleeps) == 4  # no sleep after the final attempt


def test_backoff_grows_and_is_jittered(monkeypatch):
    fake_gemini(monkeypatch, [httpx.ConnectError("boom")] * 4)
    sleeps = []
    with pytest.raises(RuntimeError):
        gemini_embedder(KEY, TASK, attempts=4, sleep=sleeps.append)(["a"])
    assert len(sleeps) == 3
    assert sleeps == sorted(sleeps) and all(2**i <= s < 2**i + 1 for i, s in enumerate(sleeps))


def test_unexpected_errors_are_not_swallowed(monkeypatch):
    """Only transport errors are transient: anything else must surface unchanged and unretried."""
    models, _ = fake_gemini(monkeypatch, [ValueError("invalid argument"), None])
    sleeps = []

    with pytest.raises(ValueError, match="invalid argument"):
        gemini_embedder(KEY, TASK, sleep=sleeps.append)(["a"])

    assert len(models.calls) == 1 and sleeps == []


def test_retries_do_not_change_cache_behaviour_or_per_batch_commits(tmp_path, monkeypatch):
    models, _ = fake_gemini(monkeypatch, [httpx.ReadError("drop"), None, None])
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    embedder = gemini_embedder(KEY, TASK, sleep=lambda s: None)

    _, new = embed_texts(["a", "b", "c"], TASK, embedder, cache, batch_size=2)
    assert new == 3 and len(models.calls) == 3  # 2 batches, the first retried once

    replay = FakeEmbedder()
    _, again = embed_texts(["a", "b", "c"], TASK, replay, cache)
    assert again == 0 and replay.batches == []  # everything committed despite the mid-run failure
