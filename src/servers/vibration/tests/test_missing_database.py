"""A wrong asset and a missing database produce different error messages."""

import pytest

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
