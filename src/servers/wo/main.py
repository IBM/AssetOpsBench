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

mcp = FastMCP("wo")

# Register tools — imported after mcp is created to avoid circular imports.
from . import tools  # noqa: E402

_TOOLS = [
    tools.get_work_orders,
    tools.get_preventive_work_orders,
    tools.get_corrective_work_orders,
    tools.get_events,
    tools.get_failure_codes,
    tools.get_work_order_distribution,
    tools.predict_next_work_order,
    tools.analyze_alert_to_failure,
]
for _fn in _TOOLS:
    mcp.tool()(_fn)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
