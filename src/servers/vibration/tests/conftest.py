"""Fixtures and helpers for vibration server tests."""

import json
import os

from dotenv import load_dotenv
import pytest
from unittest.mock import patch

load_dotenv()

# --- Custom markers ---

requires_couchdb = pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None,
    reason="CouchDB not available (set COUCHDB_URL)",
)


# --- Fixtures ---


@pytest.fixture
def mock_db():
    """Patch the module-level _get_db in couchdb_client with a mock."""
    with patch("servers.vibration.couchdb_client._get_db") as mock:
        yield mock


@pytest.fixture
def no_db():
    """Patch _get_db to return None (simulate disconnected CouchDB)."""
    with patch("servers.vibration.couchdb_client._get_db", return_value=None):
        yield


@pytest.fixture(autouse=True)
def reset_data_store():
    """Clear the in-memory data store between tests."""
    from servers.vibration.data_store import store

    store._entries.clear()
    yield
    store._entries.clear()


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    result = await mcp_instance.call_tool(tool_name, args)
    # FastMCP may return (contents, is_error) tuple or just the contents list
    if isinstance(result, tuple):
        contents = result[0]
    else:
        contents = result
    if isinstance(contents, dict):
        return contents
    return json.loads(contents[0].text)
