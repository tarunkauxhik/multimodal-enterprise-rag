import json

import httpx
import pytest

from rag.rerank import jina_reranker

KEY = "jina_secret_key_123"


def reranker(handler, sleeps):
    return jina_reranker(KEY, transport=httpx.MockTransport(handler), sleep=sleeps.append)


def test_sends_model_top_n_and_returns_sorted_pairs():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.9}]})

    result = reranker(handler, [])("q", ["d0", "d1", "d2"], 2)
    assert result == [(0, 0.9), (1, 0.2)]
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {
        "model": "jina-reranker-v3",
        "query": "q",
        "documents": ["d0", "d1", "d2"],
        "top_n": 2,
        "return_documents": False,
    }


def test_retries_rate_limit_then_succeeds():
    responses = iter([httpx.Response(429, text="slow down"), httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})])
    sleeps = []
    assert reranker(lambda r: next(responses), sleeps)("q", ["d"], 1) == [(0, 1.0)]
    assert len(sleeps) == 1


def test_client_error_is_not_retried_and_key_is_redacted():
    sleeps = []
    with pytest.raises(RuntimeError) as exc:
        reranker(lambda r: httpx.Response(401, text=f"invalid key {KEY}"), sleeps)("q", ["d"], 1)
    assert "HTTP 401" in str(exc.value) and KEY not in str(exc.value)
    assert sleeps == []


def test_transport_errors_retry_until_attempts_exhausted():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("boom")

    with pytest.raises(RuntimeError, match="ConnectError"):
        reranker(handler, sleeps)("q", ["d"], 1)
    assert len(calls) == 5 and len(sleeps) == 4
