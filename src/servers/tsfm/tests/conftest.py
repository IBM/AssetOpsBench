"""Shared fixtures and helpers for TSFM feature catalog MCP server tests."""

from __future__ import annotations

import json


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return the parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
