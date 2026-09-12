"""Client-side adaptation of the AssetOpsBench MCP tool surface.

The servers are the environment and stay frozen. This package is the only
surface an evolved harness patch may change, so it is also the only thing the
optimizer's scope gate has to police.
"""

from .apply import AdaptedTool, adapt_tool, adapt_tools, unknown_tool_names
from .spec import AdapterError, AdapterSpec, Composite, ToolAdaptation

__all__ = [
    "AdaptedTool",
    "AdapterError",
    "AdapterSpec",
    "Composite",
    "ToolAdaptation",
    "adapt_tool",
    "adapt_tools",
    "unknown_tool_names",
]
