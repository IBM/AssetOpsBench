"""Tests for FMSR MCP server tools.

Unit tests use mocked LLM chains; integration tests require live WatsonX
credentials (skipped unless WATSONX_APIKEY is set).
"""

import pytest
from servers.fmsr.main import mcp
from .conftest import call_tool, requires_watsonx


# ---------------------------------------------------------------------------
# get_failure_modes
# ---------------------------------------------------------------------------


class TestGetFailureModes:
    @pytest.mark.anyio
    async def test_chiller_returns_hardcoded(self):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "chiller"})
        assert "failure_modes" in data
        assert len(data["failure_modes"]) == 7
        assert any("Compressor" in fm for fm in data["failure_modes"])

    @pytest.mark.anyio
    async def test_chiller_number_stripped(self):
        """'Chiller 6' normalises to 'chiller' for the lookup."""
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Chiller 6"})
        assert "failure_modes" in data
        assert len(data["failure_modes"]) == 7

    @pytest.mark.anyio
    async def test_ahu_returns_hardcoded(self):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "ahu"})
        assert "failure_modes" in data
        assert len(data["failure_modes"]) == 5

    @pytest.mark.anyio
    async def test_empty_asset_name_returns_error(self):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": ""})
        assert "error" in data

    @pytest.mark.anyio
    async def test_unknown_asset_no_llm(self, no_llm):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Pump"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_unknown_asset_llm_fallback(self, mock_asset2fm_chain):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Pump"})
        assert "failure_modes" in data
        assert data["failure_modes"] == ["Fan Failure", "Belt Wear"]
        mock_asset2fm_chain.assert_called_once_with("Pump")

    @requires_watsonx
    @pytest.mark.anyio
    async def test_unknown_asset_integration(self):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Boiler"})
        assert "failure_modes" in data
        assert len(data["failure_modes"]) > 0


# ---------------------------------------------------------------------------
# get_failure_mode_sensor_mapping
# ---------------------------------------------------------------------------


_FAILURE_MODES = ["Compressor Overheating", "Condenser Water side fouling"]
_SENSORS = ["Chiller 6 Power Input", "Chiller 6 Supply Temperature"]


class TestGetFailureModeSensorMapping:
    @pytest.mark.anyio
    async def test_returns_expected_keys(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": _FAILURE_MODES, "sensors": _SENSORS},
        )
        assert "fm2sensor" in data
        assert "sensor2fm" in data
        assert "full_relevancy" in data
        assert data["metadata"]["asset_name"] == "Chiller 6"

    @pytest.mark.anyio
    async def test_full_relevancy_count(self, mock_relevancy_chain):
        """2 sensors × 2 failure modes = 4 pairs."""
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": _FAILURE_MODES, "sensors": _SENSORS},
        )
        assert len(data["full_relevancy"]) == 4

    @pytest.mark.anyio
    async def test_empty_failure_modes_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": [], "sensors": _SENSORS},
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_empty_sensors_returns_error(self, mock_relevancy_chain):
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": _FAILURE_MODES, "sensors": []},
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {"asset_name": "Chiller 6", "failure_modes": _FAILURE_MODES, "sensors": _SENSORS},
        )
        assert "error" in data

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "get_failure_mode_sensor_mapping",
            {
                "asset_name": "Chiller 6",
                "failure_modes": ["Compressor Overheating"],
                "sensors": ["Chiller 6 Power Input"],
            },
        )
        assert "full_relevancy" in data
        assert len(data["full_relevancy"]) == 1
        assert data["full_relevancy"][0]["relevancy_answer"] in ("Yes", "No", "Unknown")
