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
    async def test_transformer_returns_hardcoded(self):
        data = await call_tool(mcp, "get_failure_modes", {"asset_name": "Smart Grid Transformer"})
        assert "failure_modes" in data
        assert len(data["failure_modes"]) == 9

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

class TestInterpretDGA:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_dga_chain):
        """Test correct parsing of a mocked LLM response."""
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "fault_type" in data
        assert isinstance(data["r1"], float)
        assert isinstance(data["r2"], float)
        assert isinstance(data["r3"], float)
        assert isinstance(data["confidence"], str)
        mock_dga_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "fault_type" in data
        assert len(data["reasoning"]) > 0


class TestAssessWindingTemperature:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_winding_chain):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "thermal_status" in data
        assert isinstance(data["ageing_rate"], float)
        assert isinstance(data["alarm_active"], bool)
        assert isinstance(data["trip_active"], bool)
        mock_winding_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "thermal_status" in data
        assert len(data["recommended_action"]) > 0


class TestAssessLoadProfile:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_load_chain):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "load_mva" in data
        assert isinstance(data["load_factor_pct"], float)
        assert isinstance(data["current_imbalance_pct"], float)
        assert isinstance(data["neutral_current_flag"], bool)
        mock_load_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "load_mva" in data
        assert len(data["reasoning"]) > 0


class TestTransformerHealthIndexModel:
    VALID_FEATURES = {
        "hydrogen": 100,
        "oxygen": 10,
        "nitrogen": 200,
        "methane": 50,
        "co": 5,
        "co2": 20,
        "ethylene": 15,
        "ethane": 8,
        "acetylene": 3,
        "dbds": 0.5,
        "power_factor": 0.95,
        "interfacial_v": 15,
        "dielectric_rigidity": 30,
        "water_content": 0.2,
    }
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "",
            **self.VALID_FEATURES
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_predicts_without_llm(self, no_llm):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "Transformer1",
            **self.VALID_FEATURES
        })
        assert "asset_name" in data
        assert "health_index" in data
        assert "condition" in data

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "Transformer1",
            **self.VALID_FEATURES
        })
        assert "asset_name" in data
        assert "health_index" in data
        assert "condition" in data
        assert data["asset_name"] == "Transformer1"
        assert data["condition"] in ["Very Poor", "Poor", "Fair", "Good", "Very Good"]
