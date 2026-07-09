"""Tests for FMSR MCP server tools."""

import pytest

from servers.fmsr.main import mcp

from .conftest import call_tool, requires_watsonx


class TestGetFailureModes:
    @pytest.mark.anyio
    async def test_reads_failure_modes_from_db(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]
        assert data["exhaustive"] is False
        assert data["source"] == "synthetic sample"

    @pytest.mark.anyio
    async def test_asset_class_case_normalized(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "PUMP"})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]

    @pytest.mark.anyio
    async def test_asset_class_spacing_normalized(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "  PUMP  "})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": ""})

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_missing_asset_class_returns_guidance(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "bad-pump-1"})

        assert "error" in data
        assert "no failure_mode record for asset_class 'bad pump'" in data["error"]
        assert "Input was normalized from 'bad-pump-1'" in data["error"]
        assert "Available asset_class values include: pump" in data["error"]

    @pytest.mark.anyio
    async def test_db_unavailable_returns_error(self, monkeypatch):
        monkeypatch.setattr("servers.fmsr.main.fm_db", None)

        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert data == {"error": "CouchDB not connected"}


class TestGenerateFailureModes:
    @pytest.mark.anyio
    async def test_extends_failure_modes_from_db(
        self, fake_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "Pump", "max_modes": 5},
        )

        assert data["asset_class"] == "pump"
        assert data["known"] == ["seal leakage", "impeller wear"]
        assert data["generated"] == ["bearing wear", "motor overheating"]
        assert data["failure_modes"] == [
            "seal leakage",
            "impeller wear",
            "bearing wear",
            "motor overheating",
        ]
        assert data["source"].startswith("LLM:")
        assert "nothing was persisted" in data["message"]
        mock_failure_mode_generation.assert_called_once_with(
            "pump", ["seal leakage", "impeller wear"], 5
        )

    @pytest.mark.anyio
    async def test_generates_from_scratch_for_missing_db_record(
        self, empty_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "compressor", "max_modes": 3},
        )

        assert data["asset_class"] == "compressor"
        assert data["known"] == []
        assert data["generated"] == [
            "bearing wear",
            "seal leakage",
            "motor overheating",
        ]
        mock_failure_mode_generation.assert_called_once_with("compressor", [], 3)

    @pytest.mark.anyio
    async def test_known_argument_overrides_db(
        self, fake_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {
                "asset_class": "Pump-1",
                "known": ["bearing wear"],
                "max_modes": 10,
            },
        )

        assert data["asset_class"] == "pump"
        assert data["known"] == ["bearing wear"]
        assert data["generated"] == ["seal leakage", "motor overheating"]
        mock_failure_mode_generation.assert_called_once_with(
            "pump", ["bearing wear"], 10
        )

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, mock_failure_mode_generation):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "", "max_modes": 3},
        )

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_invalid_max_modes_returns_error(self, mock_failure_mode_generation):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "pump", "max_modes": 0},
        )

        assert data == {"error": "max_modes must be greater than 0"}

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "pump", "max_modes": 3},
        )

        assert data == {"error": "LLM unavailable"}

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {
                "asset_class": "pump",
                "known": ["seal leakage"],
                "max_modes": 2,
            },
        )

        assert "generated" in data
        assert len(data["generated"]) <= 2


_FAILURE_MODES = ["Compressor Overheating", "Condenser Water side fouling"]
_SENSORS = ["Chiller 6 Power Input", "Chiller 6 Supply Temperature"]


class TestGenerateFailureModeSensorMapping:
    @pytest.mark.anyio
    async def test_returns_expected_keys(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_class": "chiller",
                "failure_modes": _FAILURE_MODES,
                "sensors": _SENSORS,
            },
        )

        assert "fm2sensor" in data
        assert "sensor2fm" in data
        assert "full_relevancy" in data
        assert data["metadata"]["asset_class"] == "chiller"
        assert data["full_relevancy"][0]["asset_class"] == "chiller"

    @pytest.mark.anyio
    async def test_full_relevancy_count(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_class": "Chiller",
                "failure_modes": _FAILURE_MODES,
                "sensors": _SENSORS,
            },
        )

        assert len(data["full_relevancy"]) == 4

    @pytest.mark.anyio
    async def test_empty_failure_modes_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {"asset_class": "chiller", "failure_modes": [], "sensors": _SENSORS},
        )

        assert "error" in data

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_class": "",
                "failure_modes": _FAILURE_MODES,
                "sensors": _SENSORS,
            },
        )

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_empty_sensors_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {"asset_class": "chiller", "failure_modes": _FAILURE_MODES, "sensors": []},
        )

        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_class": "chiller",
                "failure_modes": _FAILURE_MODES,
                "sensors": _SENSORS,
            },
        )

        assert data == {"error": "LLM unavailable"}

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_class": "chiller",
                "failure_modes": ["Compressor Overheating"],
                "sensors": ["Chiller 6 Power Input"],
            },
        )

        assert "full_relevancy" in data
        assert len(data["full_relevancy"]) == 1
        assert data["full_relevancy"][0]["relevancy_answer"] in ("Yes", "No", "Unknown")
