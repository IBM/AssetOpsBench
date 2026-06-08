"""Work Order MCP server entry point.

The tool logic is the Maximo-derived lifecycle code in ``workorders.py`` (direct
CouchDB access — Mango `_find` / `GET` / `PUT`, no pandas). The only thing adopted
from AssetOpsBench is the *MCP server definition* convention: a single
``FastMCP("wo", instructions=...)`` instance with tools registered centrally via
``mcp.tool(title=...)(fn)`` and a ``main()`` that runs over stdio. Exposed as the
``wo-mcp-server`` entry point.

Env (AssetOpsBench-compatible): COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD,
WO_DBNAME. Set AOB_READONLY=1 to expose only the read tools.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from . import workorders as wo
from .couch import CouchClient
from .models import (
    ActualsVsPlannedResult, CostsResult, ErrorResult, KpiResult, ScheduleResult,
    TasksResult, WorkOrderItem, WorkOrderMutationResult, WorkOrderResult,
    WorkOrdersResult,
)

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)

mcp = FastMCP(
    "wo",
    instructions="Work order lifecycle for industrial assets, backed by CouchDB. Query, "
                 "create, approve, assign, close, and cancel work orders; compute KPIs, "
                 "costs, and schedules. Documents use IBM Maximo mxwo field names.",
)

_db: Optional[CouchClient] = None


def db() -> CouchClient:
    global _db
    if _db is None:
        _db = CouchClient(
            base_url=os.environ.get("COUCHDB_URL", "http://localhost:5984"),
            db=os.environ.get("WO_DBNAME", "workorder"),
            username=os.environ.get("COUCHDB_USERNAME", "admin"),
            password=os.environ.get("COUCHDB_PASSWORD", "password"),
        )
    return _db


# --------------------------------------------------------------------------- #
# MCP-facing tool functions: call the Maximo-derived logic (dict envelope), then
# convert to the typed Pydantic result model (AssetOpsBench output convention).
# All WorkOrderItem fields are Optional, so docs missing columns convert cleanly.
# --------------------------------------------------------------------------- #
def _failed(res: Dict[str, Any]) -> Optional[ErrorResult]:
    """Return an ErrorResult if the envelope reports failure, else None."""
    if not res.get("success"):
        return ErrorResult(error=res.get("error", "unknown error"))
    return None


def _mutation(res: Dict[str, Any], verb: str) -> Union[WorkOrderMutationResult, ErrorResult]:
    err = _failed(res)
    if err:
        return err
    doc = res["data"]
    return WorkOrderMutationResult(
        wonum=doc.get("wonum"), siteid=doc.get("siteid"), status=doc.get("status"),
        work_order=WorkOrderItem.model_validate(doc),
        message=f"Work order {doc.get('wonum')} {verb}.")


async def list_workorders(site_id: Optional[str] = None, status: Optional[str] = None,
                          asset_num: Optional[str] = None, priority: Optional[int] = None,
                          date_from: Optional[str] = None, date_to: Optional[str] = None,
                          page_size: int = 50, page_num: int = 1) -> Union[WorkOrdersResult, ErrorResult]:
    """List work orders with optional filters (site, status, asset, priority, dates).
    status accepts OPEN / APPROVED_PENDING; page_size=0 returns all matches in one call."""
    res = await wo.list_workorders(db(), site_id, status, asset_num, priority,
                                   date_from, date_to, page_size, page_num)
    err = _failed(res)
    if err:
        return err
    d = res["data"]
    items = [WorkOrderItem.model_validate(x) for x in d["workorders"]]
    return WorkOrdersResult(site_id=site_id, status=status, total=d["totalCount"],
                            work_orders=items, message=f"Found {d['totalCount']} work order(s).")


async def get_workorder(wonum: str, site_id: str) -> Union[WorkOrderResult, ErrorResult]:
    """Get a single work order by number and site."""
    res = await wo.get_workorder(db(), wonum, site_id)
    err = _failed(res)
    if err:
        return err
    return WorkOrderResult(work_order=WorkOrderItem.model_validate(res["data"]),
                           message=f"Work order {wonum} at {site_id}.")


async def get_workorder_tasks(wonum: str, site_id: str) -> Union[TasksResult, ErrorResult]:
    """List the child tasks of a parent work order."""
    res = await wo.get_workorder_tasks(db(), wonum, site_id)
    err = _failed(res)
    if err:
        return err
    d = res["data"]
    tasks = [WorkOrderItem.model_validate(t) for t in d["tasks"]]
    return TasksResult(parent_wonum=d["parent_wonum"], site_id=d["site_id"], total=len(tasks),
                       tasks=tasks, message=f"{len(tasks)} task(s) under {wonum}.")


async def get_workorder_costs(wonum: str, site_id: str) -> Union[CostsResult, ErrorResult]:
    """Actual labor/material/service/tool cost breakdown for a work order."""
    res = await wo.get_workorder_costs(db(), wonum, site_id)
    err = _failed(res)
    if err:
        return err
    return CostsResult.model_validate({**res["data"], "message": f"Cost breakdown for {wonum}."})


async def get_workorder_actuals_vs_planned(wonum: str, site_id: str) -> Union[ActualsVsPlannedResult, ErrorResult]:
    """Estimated vs actual hours and cost variance for a work order."""
    res = await wo.get_workorder_actuals_vs_planned(db(), wonum, site_id)
    err = _failed(res)
    if err:
        return err
    return ActualsVsPlannedResult.model_validate({**res["data"], "message": f"Actuals vs planned for {wonum}."})


async def get_workorder_kpis(site_id: str, period_months: int = 3) -> Union[KpiResult, ErrorResult]:
    """Site KPIs: totals, backlog, overdue, avg completion, priority/asset breakdowns."""
    res = await wo.get_workorder_kpis(db(), site_id, period_months)
    err = _failed(res)
    if err:
        return err
    return KpiResult.model_validate({**res["data"], "message": f"KPIs for {site_id} over {period_months} month(s)."})


async def get_schedule_calendar(site_id: str, date_from: Optional[str] = None,
                                date_to: Optional[str] = None, group_by: str = "date") -> Union[ScheduleResult, ErrorResult]:
    """Scheduled (non-terminal) work orders in a date window, bucketed by day."""
    res = await wo.get_schedule_calendar(db(), site_id, date_from, date_to, group_by)
    err = _failed(res)
    if err:
        return err
    d = res["data"]
    return ScheduleResult.model_validate({**d, "message": f"{d['total_scheduled']} scheduled at {site_id}."})


async def get_my_assigned_workorders(labor_code: str, site_id: Optional[str] = None,
                                     open_only: bool = True) -> Union[WorkOrdersResult, ErrorResult]:
    """Work orders assigned to a given technician (labor code)."""
    res = await wo.get_my_assigned_workorders(db(), labor_code, site_id, open_only)
    err = _failed(res)
    if err:
        return err
    d = res["data"]
    items = [WorkOrderItem.model_validate(x) for x in d["workorders"]]
    return WorkOrdersResult(site_id=site_id, labor_code=labor_code, total=d["totalCount"],
                            work_orders=items, message=f"{d['totalCount']} work order(s) for {labor_code}.")


async def generate_work_order(description: str, asset_num: str, site_id: str,
                              priority: int = 3, work_type: str = "CM",
                              reported_by: Optional[str] = None, location: Optional[str] = None,
                              notes: Optional[str] = None, wonum: Optional[str] = None,
                              aob_source: Optional[Dict[str, Any]] = None) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Create a work order (status WAPPR). Attach aob_source provenance (agent/trigger/evidence)."""
    res = await wo.create_workorder(db(), description=description, asset_num=asset_num,
                                    site_id=site_id, priority=priority, work_type=work_type,
                                    reported_by=reported_by, location=location, notes=notes,
                                    wonum=wonum, aob_source=aob_source)
    return _mutation(res, "created (WAPPR)")


async def update_workorder(wonum: str, site_id: str, description: Optional[str] = None,
                           priority: Optional[int] = None, location: Optional[str] = None,
                           asset_num: Optional[str] = None, notes: Optional[str] = None
                           ) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Update mutable fields on a work order."""
    res = await wo.update_workorder(db(), wonum, site_id, description, priority, location, asset_num, notes)
    return _mutation(res, "updated")


async def approve_workorder(wonum: str, site_id: str) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Approve a work order (-> APPR)."""
    return _mutation(await wo.approve_workorder(db(), wonum, site_id), "approved (APPR)")


async def assign_technician(wonum: str, site_id: str, labor_code: str, craft: Optional[str] = None,
                            start_date: Optional[str] = None, hours_planned: float = 8.0
                            ) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Assign a technician (adds a wplabor line)."""
    res = await wo.assign_technician(db(), wonum, site_id, labor_code, craft, start_date, hours_planned)
    return _mutation(res, f"assigned to {labor_code}")


async def close_workorder(wonum: str, site_id: str, actual_hours: float = 0.0,
                          failure_code: Optional[str] = None, resolution_notes: Optional[str] = None
                          ) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Close a work order (-> COMP) with actuals and resolution."""
    res = await wo.close_workorder(db(), wonum, site_id, actual_hours, failure_code, resolution_notes)
    return _mutation(res, "closed (COMP)")


async def cancel_workorder(wonum: str, site_id: str, reason: Optional[str] = None
                           ) -> Union[WorkOrderMutationResult, ErrorResult]:
    """Cancel a work order (-> CAN)."""
    return _mutation(await wo.cancel_workorder(db(), wonum, site_id, reason), "cancelled (CAN)")


# --------------------------------------------------------------------------- #
# Central registration (AssetOpsBench convention)
# --------------------------------------------------------------------------- #
_READ_TOOLS = [
    (list_workorders, "List Work Orders"),
    (get_workorder, "Get Work Order"),
    (get_workorder_tasks, "Get Work Order Tasks"),
    (get_workorder_costs, "Get Work Order Costs"),
    (get_workorder_actuals_vs_planned, "Get Work Order Actuals vs Planned"),
    (get_workorder_kpis, "Get Work Order KPIs"),
    (get_schedule_calendar, "Get Schedule Calendar"),
    (get_my_assigned_workorders, "Get My Assigned Work Orders"),
]
_WRITE_TOOLS = [
    (generate_work_order, "Generate Work Order"),
    (update_workorder, "Update Work Order"),
    (approve_workorder, "Approve Work Order"),
    (assign_technician, "Assign Technician"),
    (close_workorder, "Close Work Order"),
    (cancel_workorder, "Cancel Work Order"),
]

_TOOLS = _READ_TOOLS if os.environ.get("AOB_READONLY") == "1" else _READ_TOOLS + _WRITE_TOOLS
for _fn, _title in _TOOLS:
    mcp.tool(title=_title)(_fn)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()