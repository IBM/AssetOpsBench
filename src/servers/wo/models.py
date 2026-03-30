"""Pydantic result models for the Work Order MCP server."""

from typing import List, Optional
from pydantic import BaseModel


class ErrorResult(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Work orders
# ---------------------------------------------------------------------------


class WorkOrderItem(BaseModel):
    wo_id: str
    wo_description: str
    collection: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str
    equipment_id: str
    equipment_name: str
    preventive: bool
    work_priority: Optional[int]
    actual_finish: Optional[str]
    duration: Optional[str]
    actual_labor_hours: Optional[str]


class WorkOrdersResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total: int
    work_orders: List[WorkOrderItem]
    message: str


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventItem(BaseModel):
    event_id: str
    event_group: str
    event_category: str
    event_type: Optional[str]
    description: Optional[str]
    equipment_id: str
    equipment_name: str
    event_time: str
    note: Optional[str]


class EventsResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total: int
    events: List[EventItem]
    message: str


# ---------------------------------------------------------------------------
# Failure codes
# ---------------------------------------------------------------------------


class FailureCodeItem(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str


class FailureCodesResult(BaseModel):
    total: int
    failure_codes: List[FailureCodeItem]


# ---------------------------------------------------------------------------
# Work order distribution
# ---------------------------------------------------------------------------


class WorkOrderDistributionEntry(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str
    count: int


class WorkOrderDistributionResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_work_orders: int
    distribution: List[WorkOrderDistributionEntry]
    message: str


# ---------------------------------------------------------------------------
# Next work order prediction
# ---------------------------------------------------------------------------


class NextWorkOrderEntry(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    probability: float


class NextWorkOrderPredictionResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    last_work_order_type: str
    predictions: List[NextWorkOrderEntry]
    message: str


# ---------------------------------------------------------------------------
# Alert-to-failure analysis
# ---------------------------------------------------------------------------


class AlertToFailureEntry(BaseModel):
    transition: str
    probability: float
    average_hours_to_maintenance: Optional[float]


class AlertToFailureResult(BaseModel):
    equipment_id: str
    rule_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_alerts_analyzed: int
    transitions: List[AlertToFailureEntry]
    message: str
