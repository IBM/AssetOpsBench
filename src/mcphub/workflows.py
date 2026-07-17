"""
Composed workflows — plain functions that chain tool calls.

Each takes a ``ToolUniverse`` instance and returns a dict. Names listed in
``REGISTERED`` are auto-registered so they run through the same entrypoint.

To add one: write a function ``fn(tu, **arguments)`` and add its name to
``REGISTERED``.
"""

import re

# IoT sensor workflows are temporarily not auto-registered while the IoT MCP
# server exposes only registry discovery tools.
REGISTERED = []


def _asset_class_from_asset_id(asset_id):
    key = re.sub(r"\d+", "", asset_id or "")
    key = re.sub(r"[_\-]+", " ", key)
    return re.sub(r"\s+", " ", key).strip().lower()


def chiller_triage(tu, asset_id, site="MAIN", raise_work_order=True, priority="2"):
    """Disabled while the IoT MCP server does not expose sensor tools."""
    raise RuntimeError(
        "chiller_triage is disabled while the IoT MCP server does not expose "
        "sensor tools."
    )


def sensor_inventory_gap(tu, asset_id, site="MAIN"):
    """Disabled while the IoT MCP server does not expose sensor tools."""
    raise RuntimeError(
        "sensor_inventory_gap is disabled while the IoT MCP server does not expose "
        "sensor tools."
    )


def _names(value):
    if isinstance(value, dict):
        for key in ("sensors", "names", "data", "items"):
            if key in value:
                return _names(value[key])
        return list(value.keys())
    if isinstance(value, list):
        return [v if isinstance(v, str) else v.get("name", str(v)) for v in value]
    return [value]
