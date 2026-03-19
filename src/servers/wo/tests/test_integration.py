"""Live integration tests for the Work Order MCP server.

Requires CouchDB to be running and populated (``COUCHDB_URL`` must be set).
All tests are skipped automatically when CouchDB is unavailable.

Run with:
    uv run pytest src/servers/wo/tests/test_integration.py -v
"""

import pytest
from servers.wo.main import mcp
from .conftest import requires_couchdb, call_tool

# Real equipment IDs and rule IDs present in the sample dataset
EQUIPMENT_ID = "CWC04013"         # 431 work orders in dataset
EQUIPMENT_RICH = "CWC04014"       # 524 work orders — most records
EQUIPMENT_ALERT = "CWC04009"      # has alert events with RUL0018
RULE_ID = "RUL0018"               # 183 alert events for CWC04009


# ---------------------------------------------------------------------------
# get_work_orders — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetWorkOrdersLive:
    @pytest.mark.anyio
    async def test_returns_results(self):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        assert "work_orders" in data
        assert data["total"] > 0
        assert len(data["work_orders"]) == data["total"]

    @pytest.mark.anyio
    async def test_date_range_narrows_results(self):
        all_data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        filtered = await call_tool(
            mcp,
            "get_work_orders",
            {"equipment_id": EQUIPMENT_ID, "start_date": "2015-01-01", "end_date": "2017-12-31"},
        )
        assert filtered["total"] < all_data["total"]
        assert filtered["total"] > 0

    @pytest.mark.anyio
    async def test_each_wo_has_required_fields(self):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        required = {"wo_id", "wo_description", "equipment_id", "primary_code", "preventive", "actual_finish"}
        for wo in data["work_orders"]:
            assert required <= wo.keys()
            assert wo["equipment_id"].upper() == EQUIPMENT_ID.upper()

    @pytest.mark.anyio
    async def test_preventive_field_is_bool(self):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        for wo in data["work_orders"]:
            assert isinstance(wo["preventive"], bool)

    @pytest.mark.anyio
    async def test_unknown_equipment_returns_error(self):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "DOES_NOT_EXIST"})
        assert "error" in data


# ---------------------------------------------------------------------------
# get_preventive_work_orders — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetPreventiveWorkOrdersLive:
    @pytest.mark.anyio
    async def test_all_results_are_preventive(self):
        data = await call_tool(mcp, "get_preventive_work_orders", {"equipment_id": EQUIPMENT_ID})
        assert "work_orders" in data
        assert data["total"] > 0
        for wo in data["work_orders"]:
            assert wo["preventive"] is True

    @pytest.mark.anyio
    async def test_count_less_than_all_work_orders(self):
        all_data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        prev_data = await call_tool(mcp, "get_preventive_work_orders", {"equipment_id": EQUIPMENT_ID})
        assert prev_data["total"] <= all_data["total"]


# ---------------------------------------------------------------------------
# get_corrective_work_orders — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetCorrectiveWorkOrdersLive:
    @pytest.mark.anyio
    async def test_all_results_are_corrective(self):
        data = await call_tool(mcp, "get_corrective_work_orders", {"equipment_id": EQUIPMENT_ID})
        assert "work_orders" in data
        assert data["total"] > 0
        for wo in data["work_orders"]:
            assert wo["preventive"] is False

    @pytest.mark.anyio
    async def test_preventive_and_corrective_partition_all(self):
        all_data = await call_tool(mcp, "get_work_orders", {"equipment_id": EQUIPMENT_ID})
        prev_data = await call_tool(mcp, "get_preventive_work_orders", {"equipment_id": EQUIPMENT_ID})
        corr_data = await call_tool(mcp, "get_corrective_work_orders", {"equipment_id": EQUIPMENT_ID})
        assert prev_data["total"] + corr_data["total"] == all_data["total"]


# ---------------------------------------------------------------------------
# get_events — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetEventsLive:
    @pytest.mark.anyio
    async def test_returns_events(self):
        data = await call_tool(mcp, "get_events", {"equipment_id": EQUIPMENT_ID})
        assert "events" in data
        assert data["total"] > 0

    @pytest.mark.anyio
    async def test_event_groups_valid(self):
        data = await call_tool(mcp, "get_events", {"equipment_id": EQUIPMENT_ID})
        valid_groups = {"WORK_ORDER", "ALERT", "ANOMALY"}
        for event in data["events"]:
            assert event["event_group"] in valid_groups

    @pytest.mark.anyio
    async def test_each_event_has_required_fields(self):
        data = await call_tool(mcp, "get_events", {"equipment_id": EQUIPMENT_ID})
        required = {"event_id", "event_group", "event_category", "equipment_id", "event_time"}
        for event in data["events"]:
            assert required <= event.keys()
            assert event["equipment_id"].upper() == EQUIPMENT_ID.upper()

    @pytest.mark.anyio
    async def test_date_range_filters_events(self):
        data = await call_tool(
            mcp,
            "get_events",
            {"equipment_id": EQUIPMENT_ID, "start_date": "2015-01-01", "end_date": "2015-12-31"},
        )
        assert "events" in data
        assert data["total"] > 0
        for event in data["events"]:
            assert event["event_time"].startswith("2015")


# ---------------------------------------------------------------------------
# get_failure_codes — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetFailureCodesLive:
    @pytest.mark.anyio
    async def test_returns_codes(self):
        data = await call_tool(mcp, "get_failure_codes", {})
        assert "failure_codes" in data
        assert data["total"] > 0

    @pytest.mark.anyio
    async def test_required_fields_present(self):
        data = await call_tool(mcp, "get_failure_codes", {})
        required = {"category", "primary_code", "primary_code_description",
                    "secondary_code", "secondary_code_description"}
        for fc in data["failure_codes"]:
            assert required <= fc.keys()

    @pytest.mark.anyio
    async def test_known_code_present(self):
        data = await call_tool(mcp, "get_failure_codes", {})
        primary_codes = {fc["primary_code"] for fc in data["failure_codes"]}
        # MT010 (Oil Analysis) and MT001 (Routine Maintenance) exist in the dataset
        assert "MT010" in primary_codes
        assert "MT001" in primary_codes


# ---------------------------------------------------------------------------
# get_work_order_distribution — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestGetWorkOrderDistributionLive:
    @pytest.mark.anyio
    async def test_returns_distribution(self):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": EQUIPMENT_ID})
        assert "distribution" in data
        assert data["total_work_orders"] > 0
        assert len(data["distribution"]) > 0

    @pytest.mark.anyio
    async def test_counts_sum_to_total(self):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": EQUIPMENT_ID})
        total_from_dist = sum(e["count"] for e in data["distribution"])
        # distribution only counts entries matched in failure_codes; total_work_orders is the raw filter count
        assert total_from_dist <= data["total_work_orders"]

    @pytest.mark.anyio
    async def test_sorted_descending(self):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": EQUIPMENT_ID})
        counts = [e["count"] for e in data["distribution"]]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.anyio
    async def test_distribution_fields_present(self):
        data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": EQUIPMENT_ID})
        required = {"category", "primary_code", "primary_code_description",
                    "secondary_code", "secondary_code_description", "count"}
        for entry in data["distribution"]:
            assert required <= entry.keys()

    @pytest.mark.anyio
    async def test_date_range_reduces_total(self):
        all_data = await call_tool(mcp, "get_work_order_distribution", {"equipment_id": EQUIPMENT_RICH})
        filtered = await call_tool(
            mcp,
            "get_work_order_distribution",
            {"equipment_id": EQUIPMENT_RICH, "start_date": "2016-01-01", "end_date": "2016-12-31"},
        )
        assert filtered["total_work_orders"] < all_data["total_work_orders"]


# ---------------------------------------------------------------------------
# predict_next_work_order — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestPredictNextWorkOrderLive:
    @pytest.mark.anyio
    async def test_returns_predictions(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": EQUIPMENT_RICH})
        assert "predictions" in data
        assert "last_work_order_type" in data
        assert len(data["predictions"]) > 0

    @pytest.mark.anyio
    async def test_probabilities_sum_to_one(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": EQUIPMENT_RICH})
        if "predictions" in data:
            total = sum(p["probability"] for p in data["predictions"])
            assert abs(total - 1.0) < 1e-6

    @pytest.mark.anyio
    async def test_prediction_fields_present(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": EQUIPMENT_RICH})
        if "predictions" in data:
            required = {"category", "primary_code", "primary_code_description", "probability"}
            for pred in data["predictions"]:
                assert required <= pred.keys()

    @pytest.mark.anyio
    async def test_probabilities_between_zero_and_one(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": EQUIPMENT_RICH})
        if "predictions" in data:
            for pred in data["predictions"]:
                assert 0.0 <= pred["probability"] <= 1.0

    @pytest.mark.anyio
    async def test_unknown_equipment_returns_error(self):
        data = await call_tool(mcp, "predict_next_work_order", {"equipment_id": "DOES_NOT_EXIST"})
        assert "error" in data


# ---------------------------------------------------------------------------
# analyze_alert_to_failure — live
# ---------------------------------------------------------------------------


@requires_couchdb
class TestAnalyzeAlertToFailureLive:
    @pytest.mark.anyio
    async def test_returns_transitions(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": RULE_ID},
        )
        assert "transitions" in data
        assert data["total_alerts_analyzed"] > 0

    @pytest.mark.anyio
    async def test_probabilities_sum_to_one(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": RULE_ID},
        )
        if "transitions" in data:
            total = sum(t["probability"] for t in data["transitions"])
            assert abs(total - 1.0) < 1e-6

    @pytest.mark.anyio
    async def test_transition_fields_present(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": RULE_ID},
        )
        if "transitions" in data:
            for t in data["transitions"]:
                assert "transition" in t
                assert "probability" in t
                assert "average_hours_to_maintenance" in t

    @pytest.mark.anyio
    async def test_work_order_transition_has_avg_hours(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": RULE_ID},
        )
        if "transitions" in data:
            wo_transitions = [t for t in data["transitions"] if t["transition"] == "WORK_ORDER"]
            for t in wo_transitions:
                assert t["average_hours_to_maintenance"] is not None
                assert t["average_hours_to_maintenance"] > 0

    @pytest.mark.anyio
    async def test_unknown_rule_returns_error(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": "NONEXISTENT_RULE"},
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_result_metadata_fields(self):
        data = await call_tool(
            mcp,
            "analyze_alert_to_failure",
            {"equipment_id": EQUIPMENT_ALERT, "rule_id": RULE_ID},
        )
        assert data.get("equipment_id", "").upper() == EQUIPMENT_ALERT.upper()
        assert data.get("rule_id", "").upper() == RULE_ID.upper()
