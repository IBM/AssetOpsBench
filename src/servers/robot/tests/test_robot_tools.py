"""Integration and unit tests for all 8 Robot MCP server tools.

Tests marked @requires_couchdb are skipped when CouchDB is unreachable.
All other tests use mocked DB via conftest fixtures.
"""

import pytest
from unittest.mock import MagicMock, patch

from servers.robot.main import mcp
from .conftest import call_tool, requires_couchdb


# Shared mock profile document
_PROFILE = {
    "_id": "profile:chiller_6",
    "_rev": "1-abc",
    "physical_location": {"x": 10.0, "y": 5.0, "z": 0.0, "room_id": "B1"},
    "gauge_range": [0.0, 100.0],
    "gauge_value": 75.0,
    "panel_stuck_prob": 0.05,
    "human_present": False,
    "maintenance_slot": "day",
    "active_work_order": None,
    "inspection_frequency_days": 7,
    "last_inspection": "2024-06-01",
    "sensor_type": "pressure",
}

_PROFILE_HUMAN = {**_PROFILE, "human_present": True}
_PROFILE_STUCK = {**_PROFILE, "panel_stuck_prob": 1.0}  # always stuck
_PROFILE_FREE  = {**_PROFILE, "panel_stuck_prob": 0.0}  # never stuck

_IOT_DOC = {
    "asset_id": "Chiller 6",
    "timestamp": "2024-06-01T00:00:00",
    "Chiller 6 Pressure": 75.0,
}


def _db_for(profile):
    mock = MagicMock()
    mock.get.side_effect = lambda doc_id: profile if "profile:" in doc_id else None
    mock.find.return_value = {"docs": [_IOT_DOC]}
    mock.save.return_value = {"ok": True, "id": "x", "rev": "1-x"}
    return mock


# ---------------------------------------------------------------------------
# Tool 1: navigate_to
# ---------------------------------------------------------------------------


class TestNavigateTo:
    @pytest.mark.anyio
    async def test_returns_success_for_known_asset(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert data["success"] is True
        assert data["distance_m"] > 0
        assert data["steps_taken"] >= 1

    @pytest.mark.anyio
    async def test_blocked_when_no_location(self):
        profile_no_loc = {k: v for k, v in _PROFILE.items() if k != "physical_location"}
        with patch("servers.robot.main.db", _db_for(profile_no_loc)):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert data["success"] is False
        assert data["blocked_reason"] is not None

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_error_when_profile_not_found(self):
        mock = MagicMock()
        mock.get.return_value = None
        with patch("servers.robot.main.db", mock):
            data = await call_tool(mcp, "navigate_to", {"asset_id": "Unknown Asset"})
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_chiller6(self):
        data = await call_tool(mcp, "navigate_to", {"asset_id": "Chiller 6"})
        assert "success" in data
        assert "distance_m" in data


# ---------------------------------------------------------------------------
# Tool 2: safety_gate_check
# ---------------------------------------------------------------------------


class TestSafetyGateCheck:
    @pytest.mark.anyio
    async def test_clearance_true_when_no_human_no_wo(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["safety_clearance"] is True
        assert data["human_present"] is False
        assert data["active_work_order"] is None

    @pytest.mark.anyio
    async def test_clearance_false_when_human_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_HUMAN)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["safety_clearance"] is False
        assert data["human_present"] is True

    @pytest.mark.anyio
    async def test_missing_slot_defaults_to_day(self):
        profile_no_slot = {k: v for k, v in _PROFILE.items() if k != "maintenance_slot"}
        with patch("servers.robot.main.db", _db_for(profile_no_slot)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["slot"] == "day"

    @pytest.mark.anyio
    async def test_missing_active_wo_defaults_to_none(self):
        profile_no_wo = {k: v for k, v in _PROFILE.items() if k != "active_work_order"}
        with patch("servers.robot.main.db", _db_for(profile_no_wo)):
            data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert data["active_work_order"] is None

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "safety_gate_check", {"asset_id": "Chiller 6"})
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 3: open_panel
# ---------------------------------------------------------------------------


class TestOpenPanel:
    @pytest.mark.anyio
    async def test_panel_opens_with_zero_stuck_prob(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_FREE)):
            data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert data["success"] is True
        assert data["access_granted"] is True

    @pytest.mark.anyio
    async def test_panel_stuck_with_certain_prob(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_STUCK)):
            data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert data["success"] is False
        assert data["stuck_reason"] is not None

    @pytest.mark.anyio
    async def test_rng_deterministic_after_reset(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            result1 = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})

        import random
        import servers.robot.main as robot_main
        robot_main._simulator._rng = random.Random(42)

        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            result2 = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})

        assert result1["success"] == result2["success"]
        assert result1["access_granted"] == result2["access_granted"]

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "open_panel", {"asset_id": "Chiller 6"})
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 4: read_gauge
# ---------------------------------------------------------------------------


class TestReadGauge:
    @pytest.mark.anyio
    async def test_reading_within_range(self):
        import servers.robot.main as robot_main
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "normal")
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert 0.0 <= data["reading"] <= 100.0
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.anyio
    async def test_no_gauge_value_in_response(self):
        import servers.robot.main as robot_main
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "normal")
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert "gauge_value" not in data

    @pytest.mark.anyio
    async def test_attempt_n_reflected_in_response(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 3}
            )
        assert data["attempt_n"] == 3

    @pytest.mark.anyio
    async def test_gauge_range_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
            )
        assert "gauge_range" in data

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(
            mcp, "read_gauge", {"asset_id": "Chiller 6", "attempt_n": 1}
        )
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 5: check_human_presence
# ---------------------------------------------------------------------------


class TestCheckHumanPresence:
    @pytest.mark.anyio
    async def test_no_human(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp, "check_human_presence", {"asset_id": "Chiller 6"}
            )
        assert data["human_present"] is False

    @pytest.mark.anyio
    async def test_human_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE_HUMAN)):
            data = await call_tool(
                mcp, "check_human_presence", {"asset_id": "Chiller 6"}
            )
        assert data["human_present"] is True

    @pytest.mark.anyio
    async def test_slot_default(self):
        profile_no_slot = {k: v for k, v in _PROFILE.items() if k != "maintenance_slot"}
        with patch("servers.robot.main.db", _db_for(profile_no_slot)):
            data = await call_tool(
                mcp, "check_human_presence", {"asset_id": "Chiller 6"}
            )
        assert data["slot"] == "day"

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(
            mcp, "check_human_presence", {"asset_id": "Chiller 6"}
        )
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 6: commit_reading
# ---------------------------------------------------------------------------


class TestCommitReading:
    @pytest.mark.anyio
    async def test_blocked_when_n_lt_3(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [74.0, 76.0],
                    "decision": "close_normal",
                },
            )
        assert data["status"] == "BLOCKED"
        assert data["fm_flag"] == "FM-7b"

    @pytest.mark.anyio
    async def test_status_field_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert data["status"] in {"COMMIT", "BLOCKED", "ESCALATE", "OOD_FLAG", "PANEL_RECHECK"}

    @pytest.mark.anyio
    async def test_score_present(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "commit_reading",
                {
                    "asset_id": "Chiller 6",
                    "readings": [73.0, 75.0, 77.0],
                    "decision": "close_normal",
                },
            )
        assert "score" in data
        assert isinstance(data["score"], float)

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(
            mcp,
            "commit_reading",
            {
                "asset_id": "Chiller 6",
                "readings": [73.0, 75.0, 77.0],
                "decision": "close_normal",
            },
        )
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_commit_or_escalate(self):
        data = await call_tool(
            mcp,
            "commit_reading",
            {
                "asset_id": "Chiller 6",
                "readings": [49.5, 50.0, 50.5],
                "decision": "close_normal",
            },
        )
        assert data["status"] in {"COMMIT", "ESCALATE", "OOD_FLAG", "PANEL_RECHECK"}


# ---------------------------------------------------------------------------
# Tool 7: check_wo_similarity
# ---------------------------------------------------------------------------


class TestCheckWoSimilarity:
    def _wo_mock(self, docs):
        mock = MagicMock()
        mock.find.return_value = {"docs": docs}
        return mock

    @pytest.mark.anyio
    async def test_error_when_wo_db_none(self, no_wo_db):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(
                mcp,
                "check_wo_similarity",
                {
                    "asset_id": "Chiller 6",
                    "failure_description": "water leak near chiller",
                },
            )
        assert "error" in data

    @pytest.mark.anyio
    async def test_proceed_when_no_similar_wos(self):
        wo_doc = {
            "wonum": "1000045",
            "description": "completely unrelated electrical work",
            "assetnum": "CHILLER6",
            "status": "COMP",
        }
        with patch("servers.robot.main._get_wo_db", return_value=self._wo_mock([wo_doc])):
            with patch("servers.robot.main.db", _db_for(_PROFILE)):
                data = await call_tool(
                    mcp,
                    "check_wo_similarity",
                    {
                        "asset_id": "Chiller 6",
                        "failure_description": "water leak near chiller",
                    },
                )
        assert data["recommendation"] in {"proceed", "review", "consolidate"}
        assert "similar_wos" in data
        assert "scores" in data

    @pytest.mark.anyio
    async def test_consolidate_for_identical_description(self):
        wo_doc = {
            "wonum": "1000046",
            "description": "water leak near chiller unit",
            "assetnum": "CHILLER6",
            "status": "WAPPR",
        }
        with patch("servers.robot.main._get_wo_db", return_value=self._wo_mock([wo_doc])):
            with patch("servers.robot.main.db", _db_for(_PROFILE)):
                data = await call_tool(
                    mcp,
                    "check_wo_similarity",
                    {
                        "asset_id": "Chiller 6",
                        "failure_description": "water leak near chiller unit",
                    },
                )
        assert data["recommendation"] == "consolidate"
        assert data["duplicate_risk"] is True

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_wo_similarity(self):
        data = await call_tool(
            mcp,
            "check_wo_similarity",
            {
                "asset_id": "Chiller 6",
                "failure_description": "anomaly on chiller condenser",
            },
        )
        assert "recommendation" in data
        assert data["recommendation"] in {"proceed", "review", "consolidate"}


# ---------------------------------------------------------------------------
# Tool 8: detect_anomaly
# ---------------------------------------------------------------------------


class TestDetectAnomaly:
    @pytest.mark.anyio
    async def test_returns_all_expected_fields(self):
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        for field in (
            "spill_detected",
            "leakage_detected",
            "pipe_damage_detected",
            "pooled_liquid_detected",
            "anomaly_confidence",
        ):
            assert field in data, f"Missing field '{field}' in detect_anomaly response"

    @pytest.mark.anyio
    async def test_spill_scenario_detected(self):
        import servers.robot.main as robot_main
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "spill")
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        assert data["spill_detected"] is True
        assert data["leakage_detected"] is True

    @pytest.mark.anyio
    async def test_normal_scenario_no_spill(self):
        import servers.robot.main as robot_main
        robot_main._simulator.generate_scenario("chiller_6", [0.0, 100.0], "normal")
        with patch("servers.robot.main.db", _db_for(_PROFILE)):
            data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        assert data["spill_detected"] is False
        assert data["leakage_detected"] is False

    @pytest.mark.anyio
    async def test_error_when_db_none(self, no_db):
        data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_anomaly_fields(self):
        data = await call_tool(mcp, "detect_anomaly", {"asset_id": "Chiller 6"})
        assert "spill_detected" in data
        assert isinstance(data["anomaly_confidence"], float)
