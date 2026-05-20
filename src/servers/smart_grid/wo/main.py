"""WO MCP server — Work Order creation and management for Smart Grid transformers.

WO = Work Order. After an agent diagnoses a fault (FMSR) and forecasts
remaining life (TSFM), it creates a work order here to schedule maintenance.

Work orders are stored in-memory during a session. In production this would
write to a CMMS (Computerised Maintenance Management System) database.
Historical fault records from the dataset are pre-loaded as a read-only
reference; new work orders created by the agent are tracked separately.

Tools exposed to the LLM agent:
  list_fault_records   — browse historical fault events from the dataset
  get_fault_record     — retrieve one historical fault event
  create_work_order    — create a new maintenance work order
  list_work_orders     — list agent-created work orders (this session)
  update_work_order    — update priority, status, or assignee
  estimate_downtime    — estimate repair downtime based on fault type / severity

Data source: ``$SG_DATA_DIR/fault_records.csv`` (read-only history) +
in-memory dict (agent-created work orders, session-scoped).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pandas as pd
from mcp.server.fastmcp import FastMCP

from servers.smart_grid.base import (
    json_safe_record,
    load_asset_metadata,
    load_fault_records,
)

mcp = FastMCP("smart-grid-wo")

_fault_records: pd.DataFrame | None = None
_asset_metadata: pd.DataFrame | None = None

# Session-scoped work order store: {wo_id: dict}
_work_orders: dict[str, dict] = {}

# Downtime estimates (hours) by fault severity — derived from dataset statistics.
_DOWNTIME_ESTIMATES = {
    "low": {"min": 2, "max": 6, "typical": 4},
    "medium": {"min": 6, "max": 16, "typical": 8},
    "high": {"min": 16, "max": 48, "typical": 24},
    "critical": {"min": 48, "max": 120, "typical": 72},
}

_VALID_PRIORITIES = {"low", "medium", "high", "critical"}
_VALID_STATUSES = {"open", "in_progress", "resolved", "closed"}


def _normalize_priority(priority: str | None) -> str | None:
    if priority is None:
        return None
    return priority.strip().lower()


def _normalize_status(status: str | None) -> str | None:
    if status is None:
        return None
    return status.strip().lower()


def _get_fault_records() -> pd.DataFrame:
    global _fault_records
    if _fault_records is None:
        _fault_records = load_fault_records()
    return _fault_records


def _get_asset_metadata() -> pd.DataFrame:
    global _asset_metadata
    if _asset_metadata is None:
        _asset_metadata = load_asset_metadata()
    return _asset_metadata


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_fault_records(
    transformer_id: str | None = None,
    fault_type: str | None = None,
    maintenance_status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Browse historical fault records from the Smart Grid dataset.

    These are read-only historical events, not agent-created work orders.
    Use list_work_orders() to see work orders created in this session.

    Args:
        transformer_id:     Filter by asset, e.g. "T-018". Optional.
        fault_type:         Substring filter on fault type, e.g. "Transformer".
        maintenance_status: Filter by status ("Scheduled", "Pending", "Completed").
        limit:              Max records to return (default 20, max 100).

    Returns:
        List of fault record dicts.
    """
    df = _get_fault_records()

    if transformer_id:
        df = df[df["transformer_id"] == transformer_id]
    if fault_type:
        df = df[df["fault_type"].str.contains(fault_type, case=False, na=False)]
    if maintenance_status:
        df = df[
            df["maintenance_status"].fillna("").str.lower()
            == maintenance_status.lower()
        ]

    records = df.head(min(limit, 100)).to_dict(orient="records")
    return [json_safe_record(record) for record in records]


@mcp.tool()
def get_fault_record(fault_id: str) -> dict:
    """Retrieve a single historical fault record by its ID.

    Args:
        fault_id: e.g. "F001". Use list_fault_records() to discover IDs.

    Returns:
        Fault record dict, or an error dict if not found.
    """
    df = _get_fault_records()
    row = df[df["fault_id"] == fault_id]
    if row.empty:
        return {"error": f"Fault record '{fault_id}' not found."}
    return json_safe_record(row.iloc[0].to_dict())


@mcp.tool()
def create_work_order(
    transformer_id: str,
    issue_description: str,
    priority: str = "medium",
    fault_type: str | None = None,
    estimated_downtime_hours: float | None = None,
) -> dict:
    """Create a new maintenance work order for a transformer.

    Args:
        transformer_id:           Asset requiring maintenance, e.g. "T-016".
        issue_description:        Plain-language description of the problem.
        priority:                 One of "low", "medium", "high", "critical".
                                  Defaults to "medium".
        fault_type:               Optional fault classification for tracking,
                                  e.g. "Arc Discharge" or "Thermal Fault T3".
        estimated_downtime_hours: Override the auto-estimated downtime.

    Returns:
        Dict with keys: work_order_id, transformer_id, issue_description,
        priority, fault_type, status, estimated_downtime_hours,
        created_at, assigned_technician (null until assigned).
    """
    priority = _normalize_priority(priority) or "medium"

    if priority not in _VALID_PRIORITIES:
        return {
            "error": f"Invalid priority '{priority}'. "
            f"Must be one of: {sorted(_VALID_PRIORITIES)}"
        }

    try:
        metadata = _get_asset_metadata()
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    if transformer_id not in set(metadata["transformer_id"]):
        return {
            "error": f"Unknown transformer_id '{transformer_id}'.",
            "valid_transformer_id_source": "$SG_DATA_DIR/asset_metadata.csv",
        }

    if estimated_downtime_hours is None:
        est = _DOWNTIME_ESTIMATES.get(priority, _DOWNTIME_ESTIMATES["medium"])
        estimated_downtime_hours = est["typical"]

    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    wo = {
        "work_order_id": wo_id,
        "transformer_id": transformer_id,
        "issue_description": issue_description,
        "priority": priority,
        "fault_type": fault_type,
        "status": "open",
        "estimated_downtime_hours": estimated_downtime_hours,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "assigned_technician": None,
        "notes": [],
    }
    _work_orders[wo_id] = wo
    return wo


@mcp.tool()
def list_work_orders(
    transformer_id: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    """List work orders created by the agent in this session.

    Args:
        transformer_id: Filter by asset. Optional.
        status:         Filter by status ("open", "in_progress", "resolved", "closed").
        priority:       Filter by priority ("low", "medium", "high", "critical").

    Returns:
        List of work order dicts, newest first.
    """
    wos = list(_work_orders.values())

    if transformer_id:
        wos = [w for w in wos if w["transformer_id"] == transformer_id]
    status = _normalize_status(status)
    priority = _normalize_priority(priority)

    if status:
        wos = [w for w in wos if w["status"] == status]
    if priority:
        wos = [w for w in wos if w["priority"] == priority]

    return sorted(wos, key=lambda w: w["created_at"], reverse=True)


@mcp.tool()
def update_work_order(
    work_order_id: str,
    status: str | None = None,
    priority: str | None = None,
    assigned_technician: str | None = None,
    note: str | None = None,
) -> dict:
    """Update an existing work order's status, priority, assignee, or add a note.

    Args:
        work_order_id:       ID returned by create_work_order(), e.g. "WO-A1B2C3D4".
        status:              New status: "open", "in_progress", "resolved", "closed".
        priority:            New priority: "low", "medium", "high", "critical".
        assigned_technician: Technician identifier, e.g. "TEC-03".
        note:                Free-text note to append to the work order log.

    Returns:
        Updated work order dict, or an error dict if not found / invalid.
    """
    if work_order_id not in _work_orders:
        return {"error": f"Work order '{work_order_id}' not found in this session."}

    wo = _work_orders[work_order_id]

    status = _normalize_status(status)
    priority = _normalize_priority(priority)

    if status is not None:
        if status not in _VALID_STATUSES:
            return {
                "error": f"Invalid status '{status}'. "
                f"Must be one of: {sorted(_VALID_STATUSES)}"
            }
        wo["status"] = status

    if priority is not None:
        if priority not in _VALID_PRIORITIES:
            return {
                "error": f"Invalid priority '{priority}'. "
                f"Must be one of: {sorted(_VALID_PRIORITIES)}"
            }
        wo["priority"] = priority

    if assigned_technician is not None:
        wo["assigned_technician"] = assigned_technician

    if note:
        wo["notes"].append(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "text": note,
            }
        )

    return wo


@mcp.tool()
def estimate_downtime(
    transformer_id: str,
    severity: str,
    fault_type: str | None = None,
) -> dict:
    """Estimate the expected downtime (hours) for maintenance on a transformer.

    Estimates are derived from the Smart Grid Fault Records dataset statistics.
    The full project may replace this with a learned model.

    Args:
        transformer_id: Asset requiring maintenance, e.g. "T-019".
        severity:       Fault severity: "low", "medium", "high", or "critical".
        fault_type:     Optional fault type for context (not used in baseline
                        estimate but recorded for traceability).

    Returns:
        Dict with keys: transformer_id, severity, fault_type,
        estimated_min_hours, estimated_max_hours, estimated_typical_hours,
        source.
    """
    if severity not in _VALID_PRIORITIES:
        return {
            "error": f"Invalid severity '{severity}'. "
            f"Must be one of: {sorted(_VALID_PRIORITIES)}"
        }

    est = _DOWNTIME_ESTIMATES[severity]
    return {
        "transformer_id": transformer_id,
        "severity": severity,
        "fault_type": fault_type,
        "estimated_min_hours": est["min"],
        "estimated_max_hours": est["max"],
        "estimated_typical_hours": est["typical"],
        "source": "dataset_statistics_baseline",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — runs the FastMCP stdio JSON-RPC loop."""
    mcp.run()


if __name__ == "__main__":
    main()
