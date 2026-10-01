"""Environment forwarded to spawned MCP servers."""

from agent.runner import mcp_server_env


def test_full_env_forwards_parent(monkeypatch):
    monkeypatch.setenv("COUCHDB_URL", "http://db:5984")
    env = mcp_server_env()
    assert env["COUCHDB_URL"] == "http://db:5984"


def test_plan_execute_executor_passes_env(monkeypatch):
    from unittest.mock import MagicMock

    from agent.plan_execute.executor import Executor, _make_stdio_params

    monkeypatch.setenv("COUCHDB_URL", "http://db:5984")
    ex = Executor(MagicMock())
    assert ex._server_env["COUCHDB_URL"] == "http://db:5984"
    params = _make_stdio_params("fmsr-mcp-server", ex._server_env)
    assert params.env["COUCHDB_URL"] == "http://db:5984"
