"""Work Order MCP server entry point.

Starts a FastMCP server that exposes work-order data as tools.
Data directory is configurable via the ``WO_DATA_DIR`` environment variable
(defaults to ``src/tmp/assetopsbench/sample_data/``).
"""

import logging
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)

mcp = FastMCP("wo", instructions="Work order analytics: query work orders, events, failure codes, and predict maintenance patterns.")

# Register tools — imported after mcp is created to avoid circular imports.
from . import tools  # noqa: E402

_TOOLS = [
    (tools.get_work_orders, "Get Work Orders"),
    (tools.get_preventive_work_orders, "Get Preventive Work Orders"),
    (tools.get_corrective_work_orders, "Get Corrective Work Orders"),
    (tools.get_events, "Get Events"),
    (tools.get_failure_codes, "Get Failure Codes"),
    (tools.get_work_order_distribution, "Get Work Order Distribution"),
    (tools.predict_next_work_order, "Predict Next Work Order"),
    (tools.analyze_alert_to_failure, "Analyze Alert to Failure"),
]
for _fn, _title in _TOOLS:
    mcp.tool(title=_title)(_fn)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
