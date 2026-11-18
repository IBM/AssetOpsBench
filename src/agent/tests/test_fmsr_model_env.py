"""FMSR_MODEL_ID pinning for spawned MCP servers."""

from agent.runner import (
    FMSR_MODEL_ENV,
    fmsr_env_overrides,
    mcp_server_env,
    resolve_fmsr_model_id,
)


def test_defaults_to_agent_model(monkeypatch):
    monkeypatch.delenv(FMSR_MODEL_ENV, raising=False)
    assert resolve_fmsr_model_id("tokenrouter/MiniMax-M3") == "tokenrouter/MiniMax-M3"


def test_explicit_value_wins(monkeypatch):
    monkeypatch.setenv(FMSR_MODEL_ENV, "watsonx/meta-llama/llama-3-3-70b-instruct")
    assert (
        resolve_fmsr_model_id("tokenrouter/MiniMax-M3")
        == "watsonx/meta-llama/llama-3-3-70b-instruct"
    )


def test_empty_explicit_counts_as_unset(monkeypatch):
    monkeypatch.setenv(FMSR_MODEL_ENV, "  ")
    assert resolve_fmsr_model_id("litellm_proxy/x") == "litellm_proxy/x"


def test_pinned_even_when_equal_to_agent_model(monkeypatch):
    monkeypatch.delenv(FMSR_MODEL_ENV, raising=False)
    assert fmsr_env_overrides("m") == {FMSR_MODEL_ENV: "m"}


def test_no_model_no_override(monkeypatch):
    monkeypatch.delenv(FMSR_MODEL_ENV, raising=False)
    assert fmsr_env_overrides(None) == {}


def test_full_env_forwards_parent(monkeypatch):
    monkeypatch.setenv("COUCHDB_URL", "http://db:5984")
    monkeypatch.delenv(FMSR_MODEL_ENV, raising=False)
    env = mcp_server_env("m")
    assert env["COUCHDB_URL"] == "http://db:5984"
    assert env[FMSR_MODEL_ENV] == "m"


def test_plan_execute_executor_passes_env(monkeypatch):
    from unittest.mock import MagicMock

    from agent.plan_execute.executor import Executor, _make_stdio_params

    monkeypatch.delenv(FMSR_MODEL_ENV, raising=False)
    llm = MagicMock()
    llm.model_id = "tokenrouter/MiniMax-M3"
    ex = Executor(llm)
    assert ex._server_env[FMSR_MODEL_ENV] == "tokenrouter/MiniMax-M3"
    params = _make_stdio_params("fmsr-mcp-server", ex._server_env)
    assert params.env[FMSR_MODEL_ENV] == "tokenrouter/MiniMax-M3"
