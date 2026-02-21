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
    """Patch the module-level `db` object in main with a mock."""
    with patch("servers.iot.main.db") as mock:
        yield mock


@pytest.fixture
def no_db():
    """Patch the module-level `db` to None (simulate disconnected CouchDB)."""
    with patch("servers.iot.main.db", None):
        yield


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
