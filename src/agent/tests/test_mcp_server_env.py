"""Environment forwarded to spawned MCP servers."""

import os

from dotenv import load_dotenv

from agent.runner import LLM_CREDENTIAL_ENV_VARS, mcp_server_env


def test_full_env_forwards_parent(monkeypatch):
    monkeypatch.setenv("COUCHDB_URL", "http://db:5984")
    env = mcp_server_env()
    assert env["COUCHDB_URL"] == "http://db:5984"


def test_llm_credentials_are_withheld(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = mcp_server_env()
    for name in LLM_CREDENTIAL_ENV_VARS:
        assert env[name] == "", name


def test_blanked_credentials_survive_load_dotenv(tmp_path, monkeypatch):
    """Pin the python-dotenv behaviour mcp_server_env relies on.

    A server's load_dotenv() must not refill a credential that was blanked,
    which it would do had the name been dropped instead.
    """
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("LITELLM_API_KEY=from-dotenv\n")
    monkeypatch.setenv("LITELLM_API_KEY", "")

    load_dotenv(dotenv_file)

    assert os.environ["LITELLM_API_KEY"] == ""


def test_plan_execute_executor_passes_env(monkeypatch):
    from unittest.mock import MagicMock

    from agent.plan_execute.executor import Executor, _make_stdio_params

    monkeypatch.setenv("COUCHDB_URL", "http://db:5984")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")
    ex = Executor(MagicMock())
    params = _make_stdio_params("fmsr-mcp-server", ex._server_env)
    assert params.env["COUCHDB_URL"] == "http://db:5984"
    assert params.env["LITELLM_API_KEY"] == ""
