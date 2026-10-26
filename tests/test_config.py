import pytest

from rag import config

SECRETS = {"MINIMAX_API_KEY": "mm-secret", "GEMINI_API_KEY": "gm-secret", "JINA_API_KEY": "jn-secret"}
BASE_URL = "https://llm.example.com/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (*SECRETS, "MINIMAX_BASE_URL", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_missing_keys_named_without_values(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret")
    with pytest.raises(RuntimeError) as exc:
        config.load_settings(env_file=None)
    msg = str(exc.value)
    assert "GEMINI_API_KEY" in msg and "JINA_API_KEY" in msg
    assert "MINIMAX_API_KEY" not in msg and "mm-secret" not in msg


def test_repr_hides_secrets_and_defaults(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MINIMAX_BASE_URL", BASE_URL)
    s = config.load_settings(env_file=None)

    assert s.qdrant_url == "http://127.0.0.1:6333"
    assert s.qdrant_api_key is None
    assert not any(v in repr(s) for v in SECRETS.values())


def test_env_file_does_not_override_real_env(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nMINIMAX_API_KEY="from-file"\nGEMINI_API_KEY=from-file\nJINA_API_KEY=from-file\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setenv("MINIMAX_BASE_URL", BASE_URL)
    s = config.load_settings(env_file=env)
    assert s.minimax_api_key == "from-file"
    assert s.gemini_api_key == "from-env"


def test_missing_minimax_key_is_reported(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-secret")
    monkeypatch.setenv("JINA_API_KEY", "jn-secret")
    monkeypatch.setenv("MINIMAX_BASE_URL", BASE_URL)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        config.load_settings(env_file=None)


def test_minimax_base_url_override_drops_trailing_slash(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1/")
    assert config.load_settings(env_file=None).minimax_base_url == "https://api.minimax.io/v1"


@pytest.mark.parametrize("bad", ["api.minimax.io/v1", "ftp://gateway/v1", "https:///v1"])
def test_a_base_url_that_is_not_a_web_url_is_rejected(monkeypatch, bad):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MINIMAX_BASE_URL", bad)
    with pytest.raises(RuntimeError, match="MINIMAX_BASE_URL must be an https URL"):
        config.load_settings(env_file=None)


@pytest.mark.parametrize(
    "url",
    ["https://api.minimax.io/v1", "https://llm.example.com:8443/v1",
     "http://localhost:8080/v1", "http://127.0.0.1:4000/v1", "http://127.8.9.10/v1", "http://[::1]:8080/v1"],
)
def test_https_anywhere_and_http_only_on_loopback_are_accepted(monkeypatch, url):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MINIMAX_BASE_URL", url)
    assert config.load_settings(env_file=None).minimax_base_url == url


@pytest.mark.parametrize(
    "url",
    ["http://llm.example.com/v1", "http://api.minimax.io/v1", "http://10.0.0.5:8080/v1",
     "http://localhost.evil.com/v1", "http://127.0.0.1.nip.io/v1"],
)
def test_remote_http_is_rejected(monkeypatch, url):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MINIMAX_BASE_URL", url)
    with pytest.raises(RuntimeError, match="must use https; http is allowed only for localhost") as exc:
        config.load_settings(env_file=None)
    assert "mm-secret" not in str(exc.value)


def test_the_llm_is_minimax_m3_with_thinking_off():
    assert (config.LLM_PROVIDER, config.LLM_MODEL, config.LLM_THINKING) == ("minimax", "MiniMax-M3", False)


def test_the_base_url_is_required_and_has_no_default(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError) as exc:
        config.load_settings(env_file=None)
    assert "MINIMAX_BASE_URL" in str(exc.value) and "mm-secret" not in str(exc.value)
