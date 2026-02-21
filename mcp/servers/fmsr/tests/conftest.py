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
    """Patch _relevancy_chain so it always returns 'Yes' without calling WatsonX."""
    mock = MagicMock()
    mock.batch.return_value = [
        {"answer": "Yes", "reason": "Relevant sensor", "temporal_behavior": "Increases"}
    ] * 100
    with patch("servers.fmsr.main._relevancy_chain", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_asset2fm_chain():
    """Patch _asset2fm_chain to return a fixed failure mode list."""
    mock = MagicMock()
    mock.invoke.return_value = ["Fan Failure", "Belt Wear"]
    with patch("servers.fmsr.main._asset2fm_chain", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


