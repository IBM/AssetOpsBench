"""Pydantic result models for the WO MCP server (AssetOpsBench output convention).

These type the *public tool boundary* only — the internals in workorders.py still
return the Maximo `{success, data, metadata}` envelope; main.py converts to these
models so FastMCP advertises an output schema (like the other AssetOpsBench servers).

IMPORTANT — partial work orders:
Maximo only requires a few fields (wonum, siteid, description, status, worktype,
reportdate). Real documents routinely omit the rest. So every field here is
Optional with a default of None: a missing column becomes ``null`` in the output and
the tool returns successfully. Required-field enforcement lives at *write* time in
the CouchDB ``validate_doc_update`` design doc, not on read. ``extra="allow"`` keeps
any unmodeled field (custom domain extensions) from being dropped.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ErrorResult(BaseModel):
    error: str


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")  # never drop unmodeled fields


# --------------------------------------------------------------------------- #
# Work order item (all fields Optional → tolerant of missing columns)
# --------------------------------------------------------------------------- #
class LaborLine(_Lenient):
    laborcode: Optional[str] = None
    craft: Optional[str] = None
    laborhrs: Optional[float] = None
    startdate: Optional[str] = None


class WorkOrderItem(_Lenient):
    # Maximo mxwo header
    wonum: Optional[str] = None
    description: Optional[str] = None
    description_longdescription: Optional[str] = None
    siteid: Optional[str] = None
    orgid: Optional[str] = None
    assetnum: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    status_date: Optional[str] = None
    worktype: Optional[str] = None
    wopriority: Optional[int] = None
    reportdate: Optional[str] = None
    reportedby: Optional[str] = None
    failurecode: Optional[str] = None
    parent: Optional[str] = None
    taskid: Optional[int] = None
    lead: Optional[str] = None
    jpnum: Optional[str] = None
    schedstart: Optional[str] = None
    schedfinish: Optional[str] = None
    targstartdate: Optional[str] = None
    targcompdate: Optional[str] = None
    actstart: Optional[str] = None
    actfinish: Optional[str] = None
    # effort / cost rollups
    estlabhrs: Optional[float] = None
    actlabhrs: Optional[float] = None
    estlabcost: Optional[float] = None
    actlabcost: Optional[float] = None
    estmatcost: Optional[float] = None
    actmatcost: Optional[float] = None
    estservcost: Optional[float] = None
    actservcost: Optional[float] = None
    esttoolcost: Optional[float] = None
    acttoolcost: Optional[float] = None
    estatapprtotalcost: Optional[float] = None
    esttotalcost: Optional[float] = None
    acttotalcost: Optional[float] = None
    # child collections + benchmark extension
    wplabor: Optional[List[LaborLine]] = None
    aob_source: Optional[Dict[str, Any]] = None
    aob_asset_class: Optional[str] = None


# --------------------------------------------------------------------------- #
# Result envelopes (one per tool)
# --------------------------------------------------------------------------- #
class WorkOrdersResult(BaseModel):
    site_id: Optional[str] = None
    status: Optional[str] = None
    labor_code: Optional[str] = None
    total: int
    work_orders: List[WorkOrderItem]
    message: str


class WorkOrderResult(BaseModel):
    work_order: WorkOrderItem
    message: str


class WorkOrderMutationResult(BaseModel):
    wonum: Optional[str] = None
    siteid: Optional[str] = None
    status: Optional[str] = None
    work_order: WorkOrderItem
    message: str


class TasksResult(BaseModel):
    parent_wonum: str
    site_id: str
    total: int
    tasks: List[WorkOrderItem]
    message: str


class CostBreakdownEntry(BaseModel):
    category: str
    amount: float
    share_pct: float


class CostsResult(_Lenient):
    wonum: Optional[str] = None
    site_id: Optional[str] = None
    status: Optional[str] = None
    assetnum: Optional[str] = None
    location: Optional[str] = None
    actual_hours: Optional[float] = None
    total_cost: Optional[float] = None
    breakdown: List[CostBreakdownEntry] = []
    message: str


class VarianceEntry(BaseModel):
    estimated: Optional[float] = None
    actual: Optional[float] = None
    variance_abs: Optional[float] = None
    variance_pct: Optional[float] = None
    over_budget: Optional[bool] = None


class ActualsVsPlannedResult(_Lenient):
    wonum: Optional[str] = None
    site_id: Optional[str] = None
    status: Optional[str] = None
    worktype: Optional[str] = None
    labor_hours: Optional[VarianceEntry] = None
    labor_cost: Optional[VarianceEntry] = None
    material_cost: Optional[VarianceEntry] = None
    service_cost: Optional[VarianceEntry] = None
    tool_cost: Optional[VarianceEntry] = None
    total_cost: Optional[VarianceEntry] = None
    message: str


class AssetCount(BaseModel):
    asset: str
    count: int


class KpiResult(_Lenient):
    site_id: str
    period_months: int
    total_workorders: int
    completed: int
    backlog: int
    overdue: int
    avg_completion_hrs: float
    priority_breakdown: Dict[str, int] = {}
    top_assets_by_wo_count: List[AssetCount] = []
    message: str


class ScheduleDay(BaseModel):
    date: str
    count: int
    workorders: List[WorkOrderItem]


class ScheduleResult(_Lenient):
    site_id: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    total_scheduled: int
    by_date: Optional[List[ScheduleDay]] = None
    workorders: Optional[List[WorkOrderItem]] = None
    message: str