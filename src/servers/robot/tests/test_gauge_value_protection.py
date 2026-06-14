"""Critical invariant: gauge_value must never appear in any tool response.

9 checks:
  1. navigate_to response
  2. safety_gate_check response
  3. open_panel response
  4. read_gauge response  ← most critical
  5. check_human_presence response
  6. commit_reading response (BLOCKED path — no DB write)
  7. commit_reading response (COMMIT path)
  8. check_wo_similarity response
  9. detect_anomaly response
"""

import pytest
from unittest.mock import MagicMock, patch

from servers.robot.main import mcp
from .conftest import call_tool


# Minimal profile doc that includes gauge_value — the field that must never leak
_PROFILE_DOC = {
    "_id": "profile:chiller_6",
    "_rev": "1-abc",
    "physical_location": {"x": 10.0, "y": 5.0, "z": 0.0, "room_id": "B1"},
    "gauge_range": [0.0, 100.0],
    "gauge_value": 75.0,          # MUST NOT appear in any tool response
    "panel_stuck_prob": 0.0,      # force panel open for test repeatability
    "human_present": False,
    "maintenance_slot": "day",
    "active_work_order": None,
    "inspection_frequency_days": 7,
    "last_inspection": "2024-01-01",
    "sensor_type": "pressure",
}

_IOT_SENSOR_DOC = {
    "asset_id": "Chiller 6",
    "timestamp": "2024-06-01T00:00:00",
    "Chiller 6 Pressure": 75.0,
}


def _make_db_mock():
    mock = MagicMock()
    mock.get.side_effect = lambda doc_id: (
        _PROFILE_DOC if "profile:" in doc_id else None
    )
    mock.find.return_value = {"docs": [_IOT_SENSOR_DOC]}
    mock.save.return_value = {"ok": True, "id": "reading:chiller_6:ts", "rev": "1-x"}
    return mock


def _no_gauge_value(data: dict) -> bool:
    """Recursively check that 'gauge_value' key is absent."""
    if "gauge_value" in data:
        return False
    for v in data.values():
        if isinstance(v, dict) and not _no_gauge_value(v):
            return False
    return True


class TestGaugeValueProtection:
    @pytest.mark.anyio
    async def test_navigate_to_no_gauge_value(self):
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert _no_gauge_value(data), f"gauge_value leaked in navigate_to: {data}"

    @pytest.mark.anyio
    async def test_safety_gate_check_no_gauge_value(self):
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert _no_gauge_value(data), f"gauge_value leaked in safety_gate_check: {data}"

    @pytest.mark.anyio
    async def test_open_panel_no_gauge_value(self):
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert _no_gauge_value(data), f"gauge_value leaked in open_panel: {data}"

    @pytest.mark.anyio
    async def test_read_gauge_no_gauge_value(self):
        """Most critical check — simulator uses gauge_value internally."""
        import servers.robot.main as robot_main
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "normal")
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert _no_gauge_value(data), f"gauge_value leaked in read_gauge: {data}"
        assert "reading" in data, "read_gauge must return 'reading' field"

    @pytest.mark.anyio
    async def test_check_human_presence_no_gauge_value(self):
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(mcp, "check_human_presence", {"asset_id": "Chiller 6"})
        assert _no_gauge_value(data), f"gauge_value leaked in check_human_presence: {data}"

    @pytest.mark.anyio
    async def test_commit_reading_blocked_no_gauge_value(self):
        """BLOCKED path (N<3): no DB write, still must not expose gauge_value."""
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [74.0, 76.0],   # only 2 → BLOCKED
                    "decision": "close_normal",
                },
            )
        assert _no_gauge_value(data), f"gauge_value leaked in commit_reading (blocked): {data}"
        assert data.get("status") == "BLOCKED"

    @pytest.mark.anyio
    async def test_commit_reading_commit_response_no_gauge_value(self):
        """COMMIT path: response must not expose gauge_value."""
        mock_db = _make_db_mock()
        with patch("servers.robot.main.db", mock_db):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert _no_gauge_value(data), f"gauge_value leaked in commit_reading (commit): {data}"

    @pytest.mark.anyio
    async def test_commit_doc_written_has_no_gauge_value(self):
        """When commit occurs, the doc written to CouchDB must not have gauge_value."""
        mock_db = _make_db_mock()
        saved_docs = []
        mock_db.save.side_effect = lambda doc: (
            saved_docs.append(doc) or {"ok": True, "id": doc["_id"], "rev": "1-x"}
        )

        with patch("servers.robot.main.db", mock_db):
            await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )

        # If the verifier scored high enough to COMMIT, saved_docs will have one entry
        for doc in saved_docs:
            assert "gauge_value" not in doc, (
                f"gauge_value found in committed CouchDB doc: {doc}"
            )

    @pytest.mark.anyio
    async def test_check_wo_similarity_no_gauge_value(self):
        wo_doc = {
            "wonum": "1000045",
            "description": "Inspect chiller pressure",
            "assetnum": "CHILLER6",
            "status": "COMP",
            "reportdate": "2024-01-15",
        }
        mock_wo = MagicMock()
        mock_wo.find.return_value = {"docs": [wo_doc]}
        with patch("servers.robot.main._get_wo_db", return_value=mock_wo):
            with patch("servers.robot.main.db", _make_db_mock()):
                data = await call_tool(
                    mcp,
                    "check_wo_similarity",
                    {
                        "asset_id": "Chiller 6",
                        "failure_description": "chiller pressure anomaly",
                    },
                )
        assert _no_gauge_value(data), f"gauge_value leaked in check_wo_similarity: {data}"

    @pytest.mark.anyio
    async def test_detect_anomaly_no_gauge_value(self):
        with patch("servers.robot.main.db", _make_db_mock()):
            data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        assert _no_gauge_value(data), f"gauge_value leaked in detect_anomaly: {data}"
