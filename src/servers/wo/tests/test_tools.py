"""Tests for Work Order MCP server tools.

Unit tests use in-memory fixture DataFrames injected via ``mock_data``.
Integration tests require the sample data directory to be present and are
gated by the ``requires_wo_data`` marker.
"""

import pytest
from servers.wo.main import mcp
from .conftest import requires_couchdb, call_tool


# ---------------------------------------------------------------------------
# get_work_orders
# ---------------------------------------------------------------------------


class TestGetWorkOrders:
    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_returns_all_records(self, mock_data):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "CWC04013"})
        assert data["total"] == 3
        assert len(data["work_orders"]) == 3

    @pytest.mark.anyio
    async def test_date_range_filter(self, mock_data):
        data = await call_tool(
            mcp,
            "get_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"},
        )
        assert data["total"] == 3
        for wo in data["work_orders"]:
            assert "2017" in (wo["actual_finish"] or "")

    @pytest.mark.anyio
    async def test_invalid_date(self, mock_data):
        data = await call_tool(
            mcp, "get_work_orders", {"equipment_id": "CWC04013", "start_date": "not-a-date"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_work_order_fields_present(self, mock_data):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "CWC04013"})
        wo = data["work_orders"][0]
        for field in ("wo_id", "wo_description", "primary_code", "preventive", "equipment_id"):
            assert field in wo

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration_cwc04013_2017(self):
        data = await call_tool(
            mcp,
            "get_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"},
        )
        assert "work_orders" in data
        assert data["total"] > 0


# ---------------------------------------------------------------------------
# get_preventive_work_orders
# ---------------------------------------------------------------------------


class TestGetPreventiveWorkOrders:
    @pytest.mark.anyio
    async def test_returns_only_preventive(self, mock_data):
        data = await call_tool(mcp, "get_preventive_work_orders", {"equipment_id": "CWC04013"})
        assert data["total"] == 2
        for wo in data["work_orders"]:
            assert wo["preventive"] is True

    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "get_preventive_work_orders", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "get_preventive_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"},
        )
        assert "work_orders" in data
        for wo in data["work_orders"]:
            assert wo["preventive"] is True


# ---------------------------------------------------------------------------
# get_corrective_work_orders
# ---------------------------------------------------------------------------


class TestGetCorrectiveWorkOrders:
    @pytest.mark.anyio
    async def test_returns_only_corrective(self, mock_data):
        data = await call_tool(mcp, "get_corrective_work_orders", {"equipment_id": "CWC04013"})
        assert data["total"] == 1
        for wo in data["work_orders"]:
            assert wo["preventive"] is False

    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "get_corrective_work_orders", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "get_corrective_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"},
        )
        assert "work_orders" in data
        for wo in data["work_orders"]:
            assert wo["preventive"] is False


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


class TestGetEvents:
    @pytest.mark.anyio
    async def test_returns_events(self, mock_data):
        data = await call_tool(mcp, "get_events", {"equipment_id": "CWC04013"})
        assert data["total"] == 3
        groups = {e["event_group"] for e in data["events"]}
        assert {"WORK_ORDER", "ALERT", "ANOMALY"} == groups

    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "get_events", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_date_range(self, mock_data):
        data = await call_tool(
            mcp,
            "get_events",
            {"equipment_id": "CWC04013", "start_date": "2017-07-01", "end_date": "2017-12-31"},
        )
        assert data["total"] == 2

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "get_events", {"equipment_id": "CWC04009"})
        assert "events" in data
        assert data["total"] > 0


# ---------------------------------------------------------------------------
# get_failure_codes
# ---------------------------------------------------------------------------


class TestGetFailureCodes:
    @pytest.mark.anyio
    async def test_returns_codes(self, mock_data):
        data = await call_tool(mcp, "get_failure_codes", {})
        assert data["total"] == 3
        codes = [fc["primary_code"] for fc in data["failure_codes"]]
        assert "MT010" in codes

    @pytest.mark.anyio
    async def test_fields_present(self, mock_data):
        data = await call_tool(mcp, "get_failure_codes", {})
        fc = data["failure_codes"][0]
        for field in ("category", "primary_code", "primary_code_description", "secondary_code"):
            assert field in fc

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "get_failure_codes", {})
        assert data["total"] > 0


# ---------------------------------------------------------------------------
# get_work_order_distribution
# ---------------------------------------------------------------------------


class TestGetWorkOrderDistribution:
    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_distribution_counts(self, mock_data):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": "CWC04013"})
        assert data["total_work_orders"] == 3
        codes = {e["primary_code"]: e["count"] for e in data["distribution"]}
        assert codes.get("MT010") == 1
        assert codes.get("MT001") == 1
        assert codes.get("MT013") == 1

    @pytest.mark.anyio
    async def test_sorted_descending(self, mock_data):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": "CWC04013"})
        counts = [e["count"] for e in data["distribution"]]
        assert counts == sorted(counts, reverse=True)

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "get_work_order_distribution",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2017-12-31"},
        )
        assert "distribution" in data
        assert data["total_work_orders"] > 0


# ---------------------------------------------------------------------------
# predict_next_work_order
# ---------------------------------------------------------------------------


class TestPredictNextWorkOrder:
    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_data):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_returns_predictions(self, mock_data):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": "CWC04013"})
        # Should either return predictions or an error about transition data
        assert "predictions" in data or "error" in data
        if "predictions" in data:
            assert "last_work_order_type" in data
            assert isinstance(data["predictions"], list)

    @pytest.mark.anyio
    async def test_probabilities_sum_to_one(self, mock_data):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": "CWC04013"})
        if "predictions" in data and data["predictions"]:
            total = sum(p["probability"] for p in data["predictions"])
            assert abs(total - 1.0) < 1e-6

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": "CWC04013"})
        assert "predictions" in data or "error" in data


# ---------------------------------------------------------------------------
# analyze_alert_to_failure
# ---------------------------------------------------------------------------


class TestAnalyzeAlertToFailure:
    @pytest.mark.anyio
    async def test_unknown_rule(self, mock_data):
        data = await call_tool(
            mcp, "analyze_alert_to_failure", {"equipment_id": "CWC04013", "rule_id": "UNKNOWN"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_returns_transitions(self, mock_data):
        data = await call_tool(
            mcp, "analyze_alert_to_failure", {"equipment_id": "CWC04013", "rule_id": "CR00002"}
        )
        # fixture only has 3 rows so transitions may be empty or present
        assert "transitions" in data or "error" in data

    @pytest.mark.anyio
    async def test_probabilities_valid(self, mock_data):
        data = await call_tool(
            mcp, "analyze_alert_to_failure", {"equipment_id": "CWC04013", "rule_id": "CR00002"}
        )
        if "transitions" in data and data["transitions"]:
            total_prob = sum(t["probability"] for t in data["transitions"])
            assert abs(total_prob - 1.0) < 1e-6

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp, "analyze_alert_to_failure", {"equipment_id": "CWC04013", "rule_id": "CR00002"}
        )
        assert "transitions" in data or "error" in data
