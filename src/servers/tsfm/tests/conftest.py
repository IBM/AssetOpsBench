"""Shared fixtures and helpers for TSFM MCP server tests."""

from __future__ import annotations

import json
import os

import pytest

# Skip marker for tests that require tsfm_public + its ML dependencies.
def _tsfm_available() -> bool:
    try:
        import tsfm_public  # noqa: F401
        return True
    except ImportError:
        return False


requires_tsfm = pytest.mark.skipif(
    not _tsfm_available(),
    reason="tsfm_public not installed",
)


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return the parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
