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
        import couchdb3

        username = os.environ.get("COUCHDB_USERNAME")
        password = os.environ.get("COUCHDB_PASSWORD")
        asset_db = couchdb3.Database(
            os.environ.get("ASSET_DBNAME", "asset"),
            url=url,
            user=username,
            password=password,
            timeout=2,
        )
        registry = asset_db.find({"assetnum": "Chiller 6"}, limit=1)["docs"]
        return bool(registry)
    except Exception:
        return False


def _iot_data_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    if not url:
        return False
    try:
        import couchdb3

        username = os.environ.get("COUCHDB_USERNAME")
        password = os.environ.get("COUCHDB_PASSWORD")
        iot_db = couchdb3.Database(
            os.environ.get("IOT_DBNAME", "iot"),
            url=url,
            user=username,
            password=password,
            timeout=2,
        )
        startkey = json.dumps("iot:Chiller 6:")
        endkey = json.dumps("iot:Chiller 6;")
        rows = iot_db.all_docs(
            startkey=startkey,
            endkey=endkey,
            inclusive_end=False,
            limit=1,
        )["rows"]
        return bool(rows)
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason=(
        "CouchDB sample asset registry not available "
        "(set COUCHDB_URL and load the Chiller 6 asset profile)"
    ),
)


requires_iot_db = pytest.mark.skipif(
    not _iot_data_reachable(),
    reason=(
        "IoT sample records database not available "
        "(set COUCHDB_URL and load the Chiller 6 telemetry records)"
    ),
)


# --- Fixtures ---


@pytest.fixture
def mock_asset_db():
    """Patch the module-level `asset_db` object in main with a mock."""
    with patch("servers.iot.main.asset_db") as mock:
        yield mock


@pytest.fixture
def mock_iot_db():
    """Patch the module-level telemetry records `iot_db` object in main with a mock."""
    with patch("servers.iot.main.iot_db") as mock:
        yield mock


@pytest.fixture
def no_asset_db():
    """Patch the module-level `asset_db` to None (simulate disconnected database)."""
    with patch("servers.iot.main.asset_db", None):
        yield


@pytest.fixture
def no_iot_db():
    """Patch the module-level telemetry records `iot_db` to None."""
    with patch("servers.iot.main.iot_db", None):
        yield


@pytest.fixture(autouse=True)
def clear_iot_caches():
    """Clear module-level caches so mocked DB responses do not leak across tests."""
    import servers.iot.main as iot_main

    iot_main._registry_sites_cache = None
    iot_main._sensor_list_cache = {}
    iot_main._iot_summary_cache = {}
    yield
    iot_main._registry_sites_cache = None
    iot_main._sensor_list_cache = {}
    iot_main._iot_summary_cache = {}


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
