"""The Stirrup runner must hand its MCP servers the parent environment.

mcp.client.stdio.stdio_client applies get_default_environment() when
StdioServerParameters.env is None, and that inherits only HOME, LOGNAME, PATH,
SHELL, TERM and USER. A server launched that way never sees COUCHDB_URL and
falls back to http://localhost:5984, which is correct only when CouchDB happens
to be published there. It is wrong for any containerised or remote CouchDB, and
it fails as a connection error inside the tool rather than at startup.
"""

from __future__ import annotations

import pytest

pytest.importorskip("stirrup.tools.mcp", reason="requires stirrup[mcp]")

from agent.stirrup_agent.runner import StirrupAgentRunner


def _config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COUCHDB_URL", "http://couchdb:5984")
    monkeypatch.setenv("WO_DBNAME", "workorder")
    runner = StirrupAgentRunner(model="watsonx/test", code_enabled=False)
    return runner._build_mcp_config()


def test_every_server_receives_couchdb_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)
    # The field is mcp_servers; "mcpServers" is only its validation alias,
    # which is what runner.py passes to model_validate. Attribute access
    # uses the Python name.
    assert config.mcp_servers, "no MCP servers configured"

    for name, server in config.mcp_servers.items():
        assert server.env is not None, f"{name} would get the SDK default env"
        assert server.env.get("COUCHDB_URL") == "http://couchdb:5984", name
        assert server.env.get("WO_DBNAME") == "workorder", name


def test_the_sdk_default_would_drop_couchdb_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the SDK behaviour this guards against, so an SDK change is visible."""
    from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS, get_default_environment

    assert "COUCHDB_URL" not in DEFAULT_INHERITED_ENV_VARS
    # monkeypatch restores a COUCHDB_URL the developer already had set.
    monkeypatch.setenv("COUCHDB_URL", "http://couchdb:5984")
    assert "COUCHDB_URL" not in get_default_environment()
