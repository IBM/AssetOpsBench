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


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """The IoT server memoizes registry sites, sensor lists, and asset docs in module globals.
    Reset them before every test so a mock in one test can't leak into the next."""
    monkeypatch.setattr("servers.iot.main._registry_sites_cache", None, raising=False)
    monkeypatch.setattr("servers.iot.main._sensor_list_cache", {}, raising=False)
    monkeypatch.setattr("servers.iot.main._asset_doc_cache", {}, raising=False)
    yield


@pytest.fixture
def mock_db():
    """Patch the telemetry reading DB (`db`) with a mock."""
    with patch("servers.iot.main.db") as mock:
        yield mock


@pytest.fixture
def no_db():
    """Patch the telemetry DB to None (disconnected)."""
    with patch("servers.iot.main.db", None):
        yield


@pytest.fixture
def mock_asset_db():
    """Patch the asset-registry DB (`asset_db`) with a mock."""
    with patch("servers.iot.main.asset_db") as mock:
        yield mock


@pytest.fixture
def no_asset_db():
    """Patch the asset-registry DB to None (disconnected)."""
    with patch("servers.iot.main.asset_db", None):
        yield


@pytest.fixture
def known_main(monkeypatch):
    """Make site 'MAIN' a known site deterministically, WITHOUT hitting asset_db — seed the
    registry-sites cache. Use in unit tests so `_is_known_site('MAIN')` passes and site
    resolution never touches the network."""
    monkeypatch.setattr("servers.iot.main._registry_sites_cache", ["MAIN"])
    yield


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return the parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)