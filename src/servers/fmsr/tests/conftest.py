import json
import os

import pytest
from unittest.mock import MagicMock, patch

requires_watsonx = pytest.mark.skipif(
    os.environ.get("WATSONX_APIKEY") is None,
    reason="WatsonX not available (set WATSONX_APIKEY)",
)


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


@pytest.fixture
def no_llm():
    """Simulate missing WatsonX credentials."""
    with patch("servers.fmsr.main._llm_available", False):
        yield


@pytest.fixture
def mock_relevancy_chain():
    """Patch _call_relevancy so it always returns 'Yes' without calling the LLM."""
    mock = MagicMock(
        return_value={"answer": "Yes", "reason": "Relevant sensor", "temporal_behavior": "Increases"}
    )
    with patch("servers.fmsr.main._call_relevancy", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_asset2fm_chain():
    """Patch _call_asset2fm to return a fixed failure mode list."""
    mock = MagicMock(return_value=["Fan Failure", "Belt Wear"])
    with patch("servers.fmsr.main._call_asset2fm", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_dga_chain():
    """Patch _call_dga to return a fixed DGAInterpretationResult."""
    mock = MagicMock(
        return_value={
            "fault_type": "Partial Discharge",
            "r1": 0.1,
            "r2": 0.2,
            "r3": 0.3,
            "code": "0.1,0.2,0.3",
            "confidence": "High",
            "reasoning": "Based on gas ratios",
            "recommended_action": "Inspect insulation",
        }
    )
    with patch("servers.fmsr.main._call_dga", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_winding_chain():
    """Patch _call_winding to return a fixed WindingTemperatureResult."""
    mock = MagicMock(
        return_value={
            "thermal_status": "Normal",
            "hot_spot_rise_c": 45.0,
            "ageing_rate": 1.0,
            "alarm_active": False,
            "trip_active": False,
            "risk_level": "Low",
            "reasoning": "Within limits",
            "recommended_action": "None",
        }
    )
    with patch("servers.fmsr.main._call_winding", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_load_chain():
    """Patch _call_load to return a fixed LoadProfileResult."""
    mock = MagicMock(
        return_value={
            "load_mva": 50.0,
            "load_factor_pct": 80.0,
            "loading_status": "Normal",
            "current_imbalance_pct": 3.0,
            "neutral_current_flag": False,
            "reasoning": "Balanced load",
            "recommended_action": "None",
        }
    )
    with patch("servers.fmsr.main._call_load", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock