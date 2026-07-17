#!/usr/bin/env python3
"""Smoke-test TSFM evidence MCP tools against a real scenario telemetry file.

Usage:
    uv run python /tmp/smoke_tsfm_evidence_tools.py
    uv run python /tmp/smoke_tsfm_evidence_tools.py --dataset /abs/path/to/file.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "src").exists():
    raise SystemExit(
        "Run this script from the AssetOpsBench repo root so the local `src/` package is importable."
    )

sys.path.insert(0, str(REPO_ROOT))

from src.servers.tsfm.main import mcp
from src.servers.tsfm.io import refs


DEFAULT_DATASET = Path(
    "/Users/chathurangishyalika/AssetOpsBenchScenarioGeneration/scenarios_data/shared/iot/asset_data_1001-1025.json"
)


def print_block(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


async def call_tool(name: str, args: dict) -> dict:
    content, _ = await mcp.call_tool(name, args)
    if not content:
        return {"error": f"{name}: empty MCP content"}
    return json.loads(content[0].text)


def to_csv_pointer(dataset_path: Path) -> str:
    if dataset_path.suffix.lower() == ".csv":
        return f"file://{dataset_path}"

    raw = json.loads(dataset_path.read_text())
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            rows = raw["data"]
        elif isinstance(raw.get("records"), list):
            rows = raw["records"]
        else:
            rows = [raw]
    else:
        raise ValueError(f"Unsupported dataset JSON structure in {dataset_path}")

    df = pd.DataFrame(rows)
    refs._ensure_workdir()
    out = Path(refs.WORKDIR) / "smoke_1001_telemetry.csv"
    df.to_csv(out, index=False)
    return f"file://{out}"


def make_nan_variant(pointer: str) -> str:
    p = Path(pointer.removeprefix("file://"))
    df = pd.read_csv(p)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found for NaN injection")
    target = numeric_cols[0]
    df.loc[df.index[5:10], target] = None
    out = p.with_name("smoke_1001_telemetry_with_nans.csv")
    df.to_csv(out, index=False)
    return f"file://{out}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test TSFM evidence MCP tools")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV or JSON telemetry source to materialize as a file pointer",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    pointer = to_csv_pointer(dataset)
    nan_pointer = make_nan_variant(pointer)

    print_block("dataset_path", {"dataset": str(dataset), "pointer": pointer})
    print_block("nan_variant", {"pointer": nan_pointer})

    result = await call_tool("list_tasks", {})
    print_block("list_tasks", result)

    result = await call_tool(
        "profile_series",
        {"dataset_path": pointer, "timestamp_column": "timestamp"},
    )
    print_block("profile_series default", result)

    result = await call_tool(
        "profile_series",
        {"dataset_path": pointer, "timestamp_column": "timestamp", "channels": ["value"]},
    )
    print_block("profile_series explicit channel", result)

    result = await call_tool(
        "characterize_series",
        {"dataset_path": pointer, "timestamp_column": "timestamp"},
    )
    print_block("characterize_series default", result)

    result = await call_tool(
        "characterize_series",
        {
            "dataset_path": pointer,
            "timestamp_column": "timestamp",
            "group_rules": "vibration_temperature",
        },
    )
    print_block("characterize_series preset grouping", result)

    result = await call_tool(
        "data_quality",
        {"dataset_path": nan_pointer, "timestamp_column": "timestamp"},
    )
    print_block("data_quality with NaNs", result)

    result = await call_tool("data_quality", {"dataset_path": "", "timestamp_column": "timestamp"})
    print_block("data_quality missing dataset_path", result)


if __name__ == "__main__":
    asyncio.run(main())
