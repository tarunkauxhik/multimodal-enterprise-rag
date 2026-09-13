import pytest

from rag import config

SECRETS = {"MINIMAX_API_KEY": "mm-secret", "GEMINI_API_KEY": "gm-secret", "JINA_API_KEY": "jn-secret"}


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
    s = config.load_settings(env_file=None)
    assert s.minimax_base_url == "https://llm-gateway-azure.penpencil.guru/v1"
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
    s = config.load_settings(env_file=env)
    assert s.minimax_api_key == "from-file"
    assert s.gemini_api_key == "from-env"
