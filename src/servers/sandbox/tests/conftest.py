"""Test configuration and fixtures for sandbox MCP server tests."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Skip tests requiring container runtime if not available
requires_container = pytest.mark.skipif(
    os.environ.get("SKIP_CONTAINER_TESTS") == "1",
    reason="Container runtime not available (set SKIP_CONTAINER_TESTS=0 to enable)",
)


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


@pytest.fixture
def mock_runtime():
    """Mock container runtime to avoid actual container execution."""
    mock = MagicMock()
    
    # Mock successful execution
    mock.run_container.return_value = (0, "Hello, World!\n", "")
    
    with patch("servers.sandbox.main._get_runtime", return_value=mock):
        with patch("servers.sandbox.main._ensure_container_image"):
            yield mock


@pytest.fixture
def mock_runtime_with_error():
    """Mock container runtime that returns an error."""
    mock = MagicMock()
    
    # Mock failed execution
    mock.run_container.return_value = (1, "", "Error: Something went wrong\n")
    
    with patch("servers.sandbox.main._get_runtime", return_value=mock):
        with patch("servers.sandbox.main._ensure_container_image"):
            yield mock


@pytest.fixture
def mock_runtime_with_timeout():
    """Mock container runtime that simulates a timeout."""
    mock = MagicMock()
    
    # Mock timeout exception
    mock.run_container.side_effect = TimeoutError("Container execution timed out")
    
    with patch("servers.sandbox.main._get_runtime", return_value=mock):
        with patch("servers.sandbox.main._ensure_container_image"):
            yield mock

# Made with Bob
