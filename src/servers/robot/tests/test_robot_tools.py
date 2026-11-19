"""Unit tests for the 12 Layer 1 Spot SDK intrinsic tools.

All tests mock the CouchDB layer — no live database required.
CouchDB integration tests are guarded by @requires_couchdb.

Tools under test:
    navigate_to     open_panel     get_battery
    get_pose        list_waypoints  capture_image
    power_on        power_off      stand
    sit             dock           undock
"""

import pytest

from .conftest import call_tool, requires_couchdb
from servers.robot.main import mcp


# ===========================================================================
# navigate_to
# ===========================================================================


@pytest.mark.asyncio
async def test_navigate_to_nominal(mock_db):
    result = await call_tool(mcp, "navigate_to", {"asset_id": "Motor 01"})
    assert result["success"] is True
    assert result["asset_id"] == "Motor 01"
    assert result["distance_m"] > 0
    assert result["steps_taken"] >= 1
    assert result["blocked_reason"] is None
    assert "Motor 01" in result["message"]


@pytest.mark.asyncio
async def test_navigate_to_no_db(no_db):
    result = await call_tool(mcp, "navigate_to", {"asset_id": "Motor 01"})
    assert "error" in result
    assert "unavailable" in result["error"].lower()


@pytest.mark.asyncio
async def test_navigate_to_unknown_asset(mock_db):
    result = await call_tool(mcp, "navigate_to", {"asset_id": "NonExistentPump"})
    assert "error" in result
    assert "profile" in result["error"].lower()


@pytest.mark.asyncio
async def test_navigate_to_no_location(mock_db):
    """Profile exists but physical_location is None."""
    import copy
    from unittest.mock import patch
    from .conftest import NOMINAL_PROFILE, _make_mock_db
    profile = copy.deepcopy(NOMINAL_PROFILE)
    profile["physical_location"] = None
    with patch("servers.robot.main.db", _make_mock_db(profile=profile)):
        result = await call_tool(mcp, "navigate_to", {"asset_id": "Motor 01"})
    assert result["success"] is False
    assert result["blocked_reason"] is not None
    assert "physical_location" in result["blocked_reason"]


# ===========================================================================
# open_panel
# ===========================================================================


@pytest.mark.asyncio
async def test_open_panel_nominal(mock_db):
    result = await call_tool(mcp, "open_panel", {"asset_id": "Motor 01"})
    assert result["success"] is True
    assert result["access_granted"] is True
    assert result["stuck_reason"] is None


@pytest.mark.asyncio
async def test_open_panel_stuck(mock_db_panel_stuck):
    """FM-1: panel_stuck=True → access_granted=False, escalation required."""
    result = await call_tool(mcp, "open_panel", {"asset_id": "Motor 01"})
    assert result["success"] is False
    assert result["access_granted"] is False
    assert result["stuck_reason"] is not None
    assert "FM-1" in result["stuck_reason"]
    assert "Escalate" in result["message"]


@pytest.mark.asyncio
async def test_open_panel_no_db(no_db):
    result = await call_tool(mcp, "open_panel", {"asset_id": "Motor 01"})
    assert "error" in result


# ===========================================================================
# get_battery
# ===========================================================================


@pytest.mark.asyncio
async def test_get_battery_nominal(mock_db):
    result = await call_tool(mcp, "get_battery", {})
    assert result["low_battery"] is False
    assert result["battery_charge_pct"] == 85.0
    assert result["battery_low_threshold"] == 20.0
    assert result["battery_estimated_runtime_s"] == 5400.0
    assert "OK" in result["message"]


@pytest.mark.asyncio
async def test_get_battery_low(mock_db_low_battery):
    """FM-9: battery below threshold → low_battery=True, abort instruction."""
    result = await call_tool(mcp, "get_battery", {})
    assert result["low_battery"] is True
    assert result["battery_charge_pct"] == 18.0
    assert "LOW BATTERY" in result["message"]
    assert "Abort" in result["message"]


@pytest.mark.asyncio
async def test_get_battery_no_db(no_db):
    result = await call_tool(mcp, "get_battery", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_get_battery_missing_state_doc(mock_db):
    """robot_state doc not found → error with helpful message."""
    from unittest.mock import MagicMock, patch
    mock = MagicMock()
    mock.get.return_value = None
    with patch("servers.robot.main.db", mock):
        result = await call_tool(mcp, "get_battery", {})
    assert "error" in result
    assert "robot_state" in result["error"]


# ===========================================================================
# get_pose
# ===========================================================================


@pytest.mark.asyncio
async def test_get_pose_nominal(mock_db):
    result = await call_tool(mcp, "get_pose", {})
    assert result["localization_ok"] is True
    assert result["pose_drift_m"] == 0.0
    assert result["fault_state"] is None
    assert "OK" in result["message"]


@pytest.mark.asyncio
async def test_get_pose_localization_failure(mock_db_loc_failure):
    """FM-10: localization_ok=False, drift=1.8 m → abort instruction."""
    result = await call_tool(mcp, "get_pose", {})
    assert result["localization_ok"] is False
    assert result["pose_drift_m"] == 1.8
    assert "LOCALIZATION FAILURE" in result["message"]
    assert "Abort" in result["message"]


@pytest.mark.asyncio
async def test_get_pose_no_db(no_db):
    result = await call_tool(mcp, "get_pose", {})
    assert "error" in result


# ===========================================================================
# list_waypoints
# ===========================================================================


@pytest.mark.asyncio
async def test_list_waypoints_all(mock_db):
    result = await call_tool(mcp, "list_waypoints", {})
    assert result["total"] == 3
    assert result["active_count"] == 3
    assert all(wp["active"] for wp in result["waypoints"])
    assert "all active" in result["message"]


@pytest.mark.asyncio
async def test_list_waypoints_filtered(mock_db):
    result = await call_tool(mcp, "list_waypoints", {"asset_id": "motor_01"})
    assert result["total"] == 1
    assert result["waypoints"][0]["waypoint_id"] == "wp_motor_01_panel"


@pytest.mark.asyncio
async def test_list_waypoints_missing(mock_db_waypoint_missing):
    """FM-11: motor_01 waypoint is inactive → escalation instruction."""
    result = await call_tool(mcp, "list_waypoints", {"asset_id": "motor_01"})
    assert result["active_count"] == 0
    inactive = result["waypoints"][0]
    assert inactive["active"] is False
    assert "INACTIVE" in result["message"]
    assert "FM-11" in result["message"]


@pytest.mark.asyncio
async def test_list_waypoints_no_db(no_db):
    result = await call_tool(mcp, "list_waypoints", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_list_waypoints_no_doc(mock_db):
    """waypoints doc returns None → error with helpful message."""
    from unittest.mock import MagicMock, patch
    mock = MagicMock()
    mock.get.return_value = None
    with patch("servers.robot.main.db", mock):
        result = await call_tool(mcp, "list_waypoints", {})
    assert "error" in result
    assert "waypoints" in result["error"]


# ===========================================================================
# capture_image
# ===========================================================================


@pytest.mark.asyncio
async def test_capture_image_no_gauge_path(mock_db):
    """Nominal profile: gauge_path=None → image_available=False."""
    result = await call_tool(mcp, "capture_image", {"asset_id": "Motor 01"})
    assert result["image_available"] is False
    assert result["occlusion_flag"] is False
    assert result["gauge_path"] is None
    assert result["gauge_range"] == [0, 200]
    assert "pending" in result["message"]


@pytest.mark.asyncio
async def test_capture_image_with_gauge_path(mock_db_with_gauge_path):
    """gauge_path set → image_available=True, vlm_hint populated."""
    result = await call_tool(mcp, "capture_image", {"asset_id": "Motor 01"})
    assert result["image_available"] is True
    assert result["occlusion_flag"] is False
    assert result["gauge_path"] == "/data/gauges/motor_01/gauge_001.jpg"
    assert "vlm_hint" in result
    assert "0" in result["vlm_hint"]    # range low
    assert "200" in result["vlm_hint"]  # range high
    assert "Pass to VLM" in result["message"]


@pytest.mark.asyncio
async def test_capture_image_panel_stuck(mock_db_panel_stuck):
    """panel_stuck=True → occlusion_flag=True, image_available=False."""
    result = await call_tool(mcp, "capture_image", {"asset_id": "Motor 01"})
    assert result["occlusion_flag"] is True
    assert result["image_available"] is False
    assert "BLOCKED" in result["message"]
    assert "FM-2" in result["message"]


@pytest.mark.asyncio
async def test_capture_image_no_db(no_db):
    result = await call_tool(mcp, "capture_image", {"asset_id": "Motor 01"})
    assert "error" in result


@pytest.mark.asyncio
async def test_capture_image_unknown_asset(mock_db):
    result = await call_tool(mcp, "capture_image", {"asset_id": "GhostPump"})
    assert "error" in result


# ===========================================================================
# gauge_value isolation — no tool must return gauge_value
# ===========================================================================


@pytest.mark.asyncio
async def test_no_tool_returns_gauge_value(mock_db_with_gauge_path):
    """None of the 12 tools should return a gauge_value field."""
    tools_and_args = [
        ("navigate_to",    {"asset_id": "Motor 01"}),
        ("open_panel",     {"asset_id": "Motor 01"}),
        ("get_battery",    {}),
        ("get_pose",       {}),
        ("list_waypoints", {}),
        ("capture_image",  {"asset_id": "Motor 01"}),
        ("power_on",       {}),
        ("power_off",      {}),
        ("stand",          {}),
        ("sit",            {}),
        ("dock",           {}),
        ("undock",         {}),
    ]
    for tool_name, args in tools_and_args:
        result = await call_tool(mcp, tool_name, args)
        assert "gauge_value" not in result, (
            f"Tool '{tool_name}' leaked gauge_value field: {result}"
        )


# ===========================================================================
# power_on
# ===========================================================================


@pytest.mark.asyncio
async def test_power_on_nominal(mock_db):
    result = await call_tool(mcp, "power_on", {})
    assert result["success"] is True
    assert result["power_state"] == "POWERED_ON"
    assert result["transition"] == "power_on"
    assert result["robot_id"] == "spot_1"


@pytest.mark.asyncio
async def test_power_on_already_on(mock_db):
    """Calling power_on when already powered on returns success (idempotent)."""
    result = await call_tool(mcp, "power_on", {})
    assert result["success"] is True
    assert "already" in result["message"].lower()


@pytest.mark.asyncio
async def test_power_on_from_off(mock_db_powered_off):
    """power_on transitions from POWERED_OFF to POWERED_ON."""
    result = await call_tool(mcp, "power_on", {})
    assert result["success"] is True
    assert result["power_state"] == "POWERED_ON"
    assert "powered on" in result["message"].lower()


@pytest.mark.asyncio
async def test_power_on_no_db(no_db):
    result = await call_tool(mcp, "power_on", {})
    assert "error" in result


# ===========================================================================
# power_off
# ===========================================================================


@pytest.mark.asyncio
async def test_power_off_sitting(mock_db_sitting):
    """Sitting + powered on → power_off succeeds."""
    result = await call_tool(mcp, "power_off", {})
    assert result["success"] is True
    assert result["power_state"] == "POWERED_OFF"
    assert result["transition"] == "power_off"


@pytest.mark.asyncio
async def test_power_off_while_standing(mock_db):
    """Robot is STANDING → power_off must refuse."""
    result = await call_tool(mcp, "power_off", {})
    assert result["success"] is False
    assert "STANDING" in result["message"]
    assert "sit()" in result["message"]


@pytest.mark.asyncio
async def test_power_off_already_off(mock_db_powered_off):
    """Already powered off → idempotent success."""
    result = await call_tool(mcp, "power_off", {})
    assert result["success"] is True
    assert "already" in result["message"].lower()


@pytest.mark.asyncio
async def test_power_off_no_db(no_db):
    result = await call_tool(mcp, "power_off", {})
    assert "error" in result


# ===========================================================================
# stand
# ===========================================================================


@pytest.mark.asyncio
async def test_stand_nominal(mock_db_sitting):
    """Powered on and sitting → stand succeeds."""
    result = await call_tool(mcp, "stand", {})
    assert result["success"] is True
    assert result["stance_state"] == "STANDING"
    assert result["power_state"] == "POWERED_ON"
    assert result["transition"] == "stand"


@pytest.mark.asyncio
async def test_stand_already_standing(mock_db):
    """Already standing → idempotent success."""
    result = await call_tool(mcp, "stand", {})
    assert result["success"] is True
    assert "already" in result["message"].lower()


@pytest.mark.asyncio
async def test_stand_powered_off(mock_db_powered_off):
    """Power off → stand must refuse."""
    result = await call_tool(mcp, "stand", {})
    assert result["success"] is False
    assert "not powered on" in result["message"].lower()
    assert "power_on()" in result["message"]


@pytest.mark.asyncio
async def test_stand_no_db(no_db):
    result = await call_tool(mcp, "stand", {})
    assert "error" in result


# ===========================================================================
# sit
# ===========================================================================


@pytest.mark.asyncio
async def test_sit_nominal(mock_db):
    """Powered on and standing → sit succeeds."""
    result = await call_tool(mcp, "sit", {})
    assert result["success"] is True
    assert result["stance_state"] == "SITTING"
    assert result["transition"] == "sit"


@pytest.mark.asyncio
async def test_sit_already_sitting(mock_db_sitting):
    """Already sitting → idempotent success."""
    result = await call_tool(mcp, "sit", {})
    assert result["success"] is True
    assert "already" in result["message"].lower()


@pytest.mark.asyncio
async def test_sit_powered_off(mock_db_powered_off):
    """Power off → sit must refuse."""
    result = await call_tool(mcp, "sit", {})
    assert result["success"] is False
    assert "not powered on" in result["message"].lower()


@pytest.mark.asyncio
async def test_sit_no_db(no_db):
    result = await call_tool(mcp, "sit", {})
    assert "error" in result


# ===========================================================================
# dock
# ===========================================================================


@pytest.mark.asyncio
async def test_dock_nominal(mock_db):
    """Nominal state with dock waypoint active → dock succeeds."""
    result = await call_tool(mcp, "dock", {})
    assert result["docked"] is True
    assert result["at_charge_station"] is True
    assert result["dock_waypoint_id"] == "wp_dock_main"
    assert result["battery_charge_pct"] == 85.0


@pytest.mark.asyncio
async def test_dock_no_db(no_db):
    result = await call_tool(mcp, "dock", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_dock_no_dock_waypoint(mock_db):
    """Waypoints doc has no dock waypoint → dock fails gracefully."""
    import copy
    from unittest.mock import patch
    from .conftest import NOMINAL_ROBOT_STATE, NOMINAL_PROFILE, _make_mock_db
    wps_no_dock = {
        "_id":      "waypoints",
        "doc_type": "waypoints",
        "waypoints": [
            {"waypoint_id": "wp_motor_01_panel", "waypoint_name": "Motor 01",
             "asset_id": "motor_01", "location_description": "Plant A",
             "x": 7.2, "y": 44.8, "active": True},
        ],
    }
    with patch("servers.robot.main.db", _make_mock_db(waypoints=wps_no_dock)):
        result = await call_tool(mcp, "dock", {})
    assert result["docked"] is False
    assert result["at_charge_station"] is False
    assert "not found" in result["message"].lower()


# ===========================================================================
# undock
# ===========================================================================


@pytest.mark.asyncio
async def test_undock_nominal(mock_db):
    """Nominal battery 85% → undock succeeds."""
    result = await call_tool(mcp, "undock", {})
    assert result["docked"] is False
    assert result["at_charge_station"] is False
    assert result["battery_charge_pct"] == 85.0


@pytest.mark.asyncio
async def test_undock_low_battery(mock_db_low_battery):
    """Battery 18% (below threshold 20%) → undock refused."""
    result = await call_tool(mcp, "undock", {})
    assert result["docked"] is True
    assert result["at_charge_station"] is True
    assert "UNDOCK REFUSED" in result["message"]


@pytest.mark.asyncio
async def test_undock_no_db(no_db):
    result = await call_tool(mcp, "undock", {})
    assert "error" in result


# ===========================================================================
# Integration (live CouchDB required)
# ===========================================================================


@requires_couchdb
def test_get_battery_live():
    """Live CouchDB: get_battery returns a valid result or a clear 'not seeded' error."""
    from servers.robot.main import get_battery
    result = get_battery()
    if hasattr(result, "error"):
        assert "robot_state" in result.error
    else:
        assert result.battery_charge_pct >= 0


@requires_couchdb
def test_list_waypoints_live():
    """Live CouchDB: list_waypoints returns waypoints or a clear 'not seeded' error."""
    from servers.robot.main import list_waypoints
    result = list_waypoints()
    if hasattr(result, "error"):
        assert "waypoints" in result.error
    else:
        assert result.total >= 0
