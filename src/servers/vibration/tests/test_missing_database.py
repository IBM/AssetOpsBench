"""A wrong asset and a missing database produce different error messages."""

import couchdb3
import pytest

from servers.db_errors import DATA_UNAVAILABLE
from servers.vibration.main import mcp

from .conftest import call_tool

_ARGS = {"site_name": "MAIN", "asset_id": "Motor-X"}


@pytest.mark.anyio
async def test_wrong_asset_reports_no_sensors(mock_db):
    mock_db.return_value.find.return_value = {"docs": []}
    mock_db.return_value.check.return_value = True

    data = await call_tool(mcp, "list_vibration_sensors", _ARGS)

    assert data["error"] == "No sensors found for asset 'Motor-X' at site 'MAIN'."


@pytest.mark.anyio
async def test_missing_database_reports_unavailable(mock_db):
    mock_db.return_value.find.side_effect = RuntimeError("Database does not exist.")
    mock_db.return_value.check.return_value = False

    data = await call_tool(mcp, "list_vibration_sensors", _ARGS)

    assert "does not exist or is unreachable" in data["error"]
    assert "vibration" not in data["error"]


@pytest.mark.anyio
async def test_unreachable_couchdb_hides_database_name_and_host(mock_db):
    # A real client against a closed port: CouchDB is unreachable.
    mock_db.return_value = couchdb3.Database("secret_vib", url="http://127.0.0.1:9")

    for tool, args in [
        ("list_vibration_sensors", _ARGS),
        ("get_vibration_data", {**_ARGS, "sensor_name": "v", "start": "2025-01-01"}),
    ]:
        data = await call_tool(mcp, tool, args)
        assert data["error"] == DATA_UNAVAILABLE, tool
