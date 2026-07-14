import json
import os

from dotenv import load_dotenv
import pytest
from unittest.mock import patch

load_dotenv()

# --- Custom markers ---


def _couchdb_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    iot_dbname = os.environ.get("IOT_DBNAME")
    if not url or not iot_dbname:
        return False
    try:
        import couchdb3

        username = os.environ.get("COUCHDB_USERNAME")
        password = os.environ.get("COUCHDB_PASSWORD")
        iot_db = couchdb3.Database(
            iot_dbname,
            url=url,
            user=username,
            password=password,
        )
        asset_db = couchdb3.Database(
            os.environ.get("ASSET_DBNAME", "asset"),
            url=url,
            user=username,
            password=password,
        )
        telemetry = iot_db.find({"asset_id": "Chiller 6"}, limit=1)["docs"]
        registry = asset_db.find(
            {"doctype": "asset", "iot_asset_id": "Chiller 6"}, limit=1
        )["docs"]
        return bool(telemetry and registry)
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason=(
        "CouchDB sample IoT data not available "
        "(set COUCHDB_URL/IOT_DBNAME and load the Chiller 6 sample data)"
    ),
)


# --- Fixtures ---


@pytest.fixture
def mock_db():
    """Patch the module-level `db` object in main with a mock."""
    with patch("servers.iot.main.db") as mock:
        yield mock


@pytest.fixture
def mock_asset_db():
    """Patch the module-level `asset_db` object in main with a mock."""
    with patch("servers.iot.main.asset_db") as mock:
        yield mock


@pytest.fixture
def no_db():
    """Patch the module-level `db` to None (simulate disconnected CouchDB)."""
    with patch("servers.iot.main.db", None):
        yield


@pytest.fixture
def no_asset_db():
    """Patch the module-level `asset_db` to None (simulate disconnected CouchDB)."""
    with patch("servers.iot.main.asset_db", None):
        yield


@pytest.fixture(autouse=True)
def clear_iot_caches():
    """Clear module-level caches so mocked DB responses do not leak across tests."""
    import servers.iot.main as iot_main

    iot_main._asset_list_cache = None
    iot_main._sensor_list_cache.clear()
    iot_main._asset_doc_cache.clear()
    iot_main._registry_sites_cache = None
    yield
    iot_main._asset_list_cache = None
    iot_main._sensor_list_cache.clear()
    iot_main._asset_doc_cache.clear()
    iot_main._registry_sites_cache = None


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
