import json
import os

from dotenv import load_dotenv
import pytest
from unittest.mock import patch

load_dotenv()

# --- Custom markers ---


def _couchdb_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    if not url:
        return False
    try:
        import requests
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason="CouchDB not reachable (set COUCHDB_URL and ensure CouchDB is running)",
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

@pytest.fixture(autouse=True)
def clear_asset_cache():
    """Clear the asset list cache between tests to prevent mock data leaking."""
    import servers.iot.main as iot_main
    iot_main._asset_list_cache = None
    iot_main._sensor_list_cache = {}
    yield
    iot_main._asset_list_cache = None
    iot_main._sensor_list_cache = {}