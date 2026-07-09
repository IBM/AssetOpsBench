"""Tests for FMSR MCP server tools."""

import pytest

from servers.fmsr.main import mcp

from .conftest import call_tool, requires_watsonx


class TestGetFailureModes:
    @pytest.mark.anyio
    async def test_reads_failure_modes_from_db(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "chiller"})

        assert data["failure_modes"] == ["Compressor overheating", "Condenser fouling"]
        assert data["exhaustive"] is False
        assert data["source"] == "test"

    @pytest.mark.anyio
    async def test_chiller_number_stripped(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Chiller 6"})

        assert data["failure_modes"] == ["Compressor overheating", "Condenser fouling"]

    @pytest.mark.anyio
    async def test_empty_asset_name_returns_error(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": ""})

        assert "error" in data

    @pytest.mark.anyio
    async def test_missing_asset_returns_error(self, empty_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "unknown"})

        assert "error" in data
        assert "try generate_failure_modes" in data["error"]

    @pytest.mark.anyio
    async def test_db_unavailable_returns_error(self, monkeypatch):
        monkeypatch.setattr("servers.fmsr.main.fm_db", None)

        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "pump"})

        assert data == {"error": "CouchDB not connected"}


class TestGenerateFailureModes:
    @pytest.mark.anyio
    async def test_generate_from_scratch(self, empty_fm_db, mock_asset2fm_chain):
        data = await call_tool(mcp, "generate_failure_modes", {"asset_name": "Pump"})

        assert data["known"] == []
        assert data["generated"] == ["Fan Failure", "Belt Wear"]
        assert data["failure_modes"] == ["Fan Failure", "Belt Wear"]
        mock_asset2fm_chain.assert_called_once_with("Pump")

    @pytest.mark.anyio
    async def test_generate_extends_known_modes(
        self, fake_fm_db, mock_asset2fm_extend_chain
    ):
        data = await call_tool(mcp, "generate_failure_modes", {"asset_name": "Pump 1"})

        assert data["known"] == ["seal leakage"]
        assert data["generated"] == ["Bearing Wear"]
        assert data["failure_modes"] == ["seal leakage", "Bearing Wear"]
        mock_asset2fm_extend_chain.assert_called_once_with("Pump 1", ["seal leakage"])

    @pytest.mark.anyio
    async def test_generate_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "generate_failure_modes", {"asset_name": "Pump"})

        assert data == {"error": "LLM unavailable"}


class TestAddFailureModes:
    @pytest.mark.anyio
    async def test_add_failure_modes_merges_existing(self, fake_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "Pump 1",
                "failure_modes": ["seal leakage", "Bearing Wear"],
                "source": "unit-test",
            },
        )

        assert data["asset_class"] == "pump"
        assert data["added"] == ["Bearing Wear"]
        assert data["total"] == 2
        assert fake_fm_db.docs["fm:pump"]["failure_modes"] == [
            "Bearing Wear",
            "seal leakage",
        ]
        assert fake_fm_db.docs["fm:pump"]["source"] == "unit-test"

    @pytest.mark.anyio
    async def test_add_failure_modes_creates_new_doc(self, empty_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "Gearbox",
                "failure_modes": ["Gear tooth wear"],
                "exhaustive": True,
            },
        )

        assert data["asset_class"] == "gearbox"
        assert data["added"] == ["Gear tooth wear"]
        assert data["total"] == 1
        assert empty_fm_db.docs["fm:gearbox"]["exhaustive"] is True


_FAILURE_MODES = ["Compressor Overheating", "Condenser Water side fouling"]
_SENSORS = ["Chiller 6 Power Input", "Chiller 6 Supply Temperature"]


class TestGenerateFailureModeSensorMapping:
    @pytest.mark.anyio
    async def test_returns_expected_keys(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_name": "Chiller 6",
                "failure_modes": _FAILURE_MODES,
                "sensors": _SENSORS,
            },
        )

        assert "fm2sensor" in data
        assert "sensor2fm" in data
        assert "full_relevancy" in data
        assert data["metadata"]["asset_name"] == "Chiller 6"

    @pytest.mark.anyio
    async def test_full_relevancy_count(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_name": "Chiller 6",
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
            {"asset_name": "Chiller 6", "failure_modes": [], "sensors": _SENSORS},
        )

        assert "error" in data

    @pytest.mark.anyio
    async def test_empty_sensors_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": _FAILURE_MODES, "sensors": []},
        )

        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(
            mcp,
            "generate_failure_mode_sensor_mapping",
            {
                "asset_name": "Chiller 6",
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
                "asset_name": "Chiller 6",
                "failure_modes": ["Compressor Overheating"],
                "sensors": ["Chiller 6 Power Input"],
            },
        )

        assert "full_relevancy" in data
        assert len(data["full_relevancy"]) == 1
        assert data["full_relevancy"][0]["relevancy_answer"] in ("Yes", "No", "Unknown")
