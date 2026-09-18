#!/usr/bin/env python3
"""Reachability and tool-surface check for the AssetOpsBench MCP servers.

This is the MCP equivalent of the "minimal import check" that a Python
repository skill would carry. It completes the stdio handshake, calls
``tools/list``, and asserts that the tool names the skill documents are the
tool names the server actually exposes.

Usage
-----
    python check_servers.py                 # all six servers
    python check_servers.py --server iot    # one server
    python check_servers.py --json          # machine-readable result

Exit codes
----------
    0  every checked server passed
    1  at least one server failed the handshake or the surface assertion
    2  the MCP client library is unavailable

Outcome vocabulary matches the DisCo native-check classes:
PASS, SKILL_GAP, NATIVE_FAIL, SKIP_UNSAFE.

    PASS        handshake succeeded and the expected tools are all present
    SKILL_GAP   handshake succeeded but the documented surface disagrees with
                the live surface; the skill is stale, not the server
    NATIVE_FAIL the server could not be launched or did not complete the
                handshake
    SKIP_UNSAFE the server was not selected for this run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Expected surface, as documented in references/mcp-servers.md.
# Work-order write tools are conditional on AOB_READONLY.
EXPECTED: dict[str, list[str]] = {
    "iot": [
        "sites", "asset_ids", "asset_detail", "measured_sensors",
        "installed_sensors", "assets", "find_assets_by_sensors",
        "stream_extent", "history", "latest_reading", "sensor_coverage",
        "sensor_stats",
    ],
    "fmsr": ["get_failure_modes", "generate_failure_modes", "add_failure_modes"],
    "vibration": [
        "get_vibration_data", "list_vibration_sensors", "compute_fft_spectrum",
        "compute_envelope_spectrum", "assess_vibration_severity",
        "calculate_bearing_frequencies", "list_known_bearings",
        "diagnose_vibration",
    ],
    "utilities": [
        "json_reader", "get_sensor_catalog", "get_asset_catalog",
        "get_failure_mode_catalog", "current_date_time", "current_time_english",
    ],
    "wo": [
        "list_workorders", "get_workorder", "get_workorder_tasks",
        "get_workorder_costs", "get_workorder_actuals_vs_planned",
        "get_workorder_kpis", "get_schedule_calendar",
        "get_my_assigned_workorders", "get_failure_codes",
    ],
    # tsfm exposes 41 tools; assert a representative spine rather than all of
    # them, so a catalog addition upstream does not read as a regression.
    "tsfm": [
        "list_tasks", "profile_series", "data_quality", "list_models",
        "find_models", "extract_features", "run_recipe", "run_plan",
        "list_results", "list_runs",
    ],
}

WO_WRITE = [
    "generate_work_order", "update_workorder", "approve_workorder",
    "assign_technician", "close_workorder", "cancel_workorder",
]

LAUNCH = {name: ["uv", "run", f"{name}-mcp-server"] for name in EXPECTED}


async def check_one(name: str, timeout: float) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    expected = list(EXPECTED[name])
    if name == "wo" and os.environ.get("AOB_READONLY") != "1":
        expected += WO_WRITE

    params = StdioServerParameters(
        command=LAUNCH[name][0], args=LAUNCH[name][1:], env=dict(os.environ)
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the failure class is the result
        return {
            "server": name,
            "status": "NATIVE_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "tools_found": 0,
        }

    found = sorted(t.name for t in listed.tools)
    missing = sorted(set(expected) - set(found))
    status = "PASS" if not missing else "SKILL_GAP"
    return {
        "server": name,
        "status": status,
        "tools_found": len(found),
        "missing_expected": missing,
        "unexpected_extra": sorted(set(found) - set(expected)) if name != "tsfm" else [],
    }


async def main_async(servers: list[str], timeout: float) -> list[dict]:
    return [await check_one(name, timeout) for name in servers]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="append", choices=sorted(EXPECTED),
                        help="check one server; repeatable; default is all six")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        import mcp  # noqa: F401
    except ImportError:
        print("mcp client library not importable; install the project first",
              file=sys.stderr)
        return 2

    servers = args.server or sorted(EXPECTED)
    results = asyncio.run(main_async(servers, args.timeout))
    skipped = [{"server": s, "status": "SKIP_UNSAFE"}
               for s in sorted(EXPECTED) if s not in servers]

    if args.json:
        print(json.dumps({"results": results + skipped}, indent=2))
    else:
        for r in results:
            line = f"{r['status']:<12} {r['server']:<10} tools={r['tools_found']}"
            if r.get("missing_expected"):
                line += f"  missing={','.join(r['missing_expected'])}"
            if r.get("error"):
                line += f"  {r['error']}"
            print(line)
        for r in skipped:
            print(f"{r['status']:<12} {r['server']}")

    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
