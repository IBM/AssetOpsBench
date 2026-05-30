"""Tool handler functions for the Work Order MCP server.

Each function is a plain Python callable.  ``main.py`` registers them on the
``FastMCP`` instance with ``mcp.tool()(fn)`` so that tests can import either
the raw functions or the decorated ``mcp`` without circular-import issues.
"""

from collections import Counter
from typing import List, Optional, Union

import pandas as pd

from .data import (
    date_conditions,
    fetch_work_orders,
    filter_df,
    get_transition_matrix,
    load,
    parse_date,
    row_to_event,
)
from .models import (
    AlertToFailureEntry,
    AlertToFailureResult,
    ErrorResult,
    EventsResult,
    FailureCodeItem,
    FailureCodesResult,
    NextWorkOrderEntry,
    NextWorkOrderPredictionResult,
    WorkOrderDistributionEntry,
    WorkOrderDistributionResult,
    WorkOrdersResult,
)


def get_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve all work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = fetch_work_orders(df, equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} work orders for '{equipment_id}'.",
    )


def get_preventive_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve only preventive work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = fetch_work_orders(df[df["preventive"] == "TRUE"], equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No preventive work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} preventive work orders for '{equipment_id}'.",
    )


def get_corrective_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve only corrective work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = fetch_work_orders(df[df["preventive"] == "FALSE"], equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No corrective work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} corrective work orders for '{equipment_id}'.",
    )


def get_events(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[EventsResult, ErrorResult]:
    """Retrieve all events (work orders, alerts, anomalies) for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = load("events")
    if df is None:
        return ErrorResult(error="Event data not available")
    try:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower()
    }
    if start_dt or end_dt:
        cond["event_time"] = lambda x, s=start_dt, e=end_dt: (
            (s is None or x >= s) and (e is None or x <= e)
        )

    filtered = filter_df(df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No events found for equipment_id '{equipment_id}'")

    events = [row_to_event(row) for _, row in filtered.iterrows()]
    return EventsResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(events),
        events=events,
        message=f"Found {len(events)} events for '{equipment_id}'.",
    )


def get_failure_codes() -> Union[FailureCodesResult, ErrorResult]:
    """Retrieve all available failure codes with their categories and descriptions."""
    df = load("failure_codes")
    if df is None:
        return ErrorResult(error="Failure codes data not available")

    items = [
        FailureCodeItem(
            category=str(row.get("category", "")),
            primary_code=str(row.get("primary_code", "")),
            primary_code_description=str(row.get("primary_code_description", "")),
            secondary_code=str(row.get("secondary_code", "")),
            secondary_code_description=str(row.get("secondary_code_description", "")),
        )
        for _, row in df.iterrows()
    ]
    return FailureCodesResult(total=len(items), failure_codes=items)


def get_work_order_distribution(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrderDistributionResult, ErrorResult]:
    """Calculate the distribution of work order types (by failure code) for a specific equipment.

    Returns counts per (primary_code, secondary_code) pair, sorted by frequency descending.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    wo_df = load("wo_events")
    fc_df = load("failure_codes")
    if wo_df is None:
        return ErrorResult(error="Work order data not available")
    if fc_df is None:
        return ErrorResult(error="Failure codes data not available")

    try:
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    filtered = wo_df[wo_df["equipment_id"] == equipment_id].copy()
    if start_dt:
        filtered = filtered[filtered["actual_finish"] >= start_dt]
    if end_dt:
        filtered = filtered[filtered["actual_finish"] <= end_dt]

    if filtered.empty:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")

    counts = (
        filtered.groupby(["primary_code", "secondary_code"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    distribution: List[WorkOrderDistributionEntry] = []
    for _, row in counts.iterrows():
        match = fc_df[
            (fc_df["primary_code"] == row["primary_code"])
            & (fc_df["secondary_code"] == row["secondary_code"])
        ]
        if match.empty:
            continue
        m = match.iloc[0]
        distribution.append(
            WorkOrderDistributionEntry(
                category=str(m.get("category", "")),
                primary_code=str(m.get("primary_code", "")),
                primary_code_description=str(m.get("primary_code_description", "")),
                secondary_code=str(m.get("secondary_code", "")),
                secondary_code_description=str(m.get("secondary_code_description", "")),
                count=int(row["count"]),
            )
        )

    return WorkOrderDistributionResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total_work_orders=int(filtered.shape[0]),
        distribution=distribution,
        message=f"Distribution across {len(distribution)} failure code(s) for '{equipment_id}'.",
    )


def predict_next_work_order(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[NextWorkOrderPredictionResult, ErrorResult]:
    """Predict the probabilities of the next expected work order types based on historical transition patterns.

    Uses a Markov-chain transition matrix built from the sequence of past work order
    primary codes to estimate what type of work order is likely to follow the most
    recent one.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    wo_df = load("wo_events")
    pfc_df = load("primary_failure_codes")
    if wo_df is None:
        return ErrorResult(error="Work order data not available")

    try:
        parse_date(start_date)
        parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond = date_conditions(equipment_id, "actual_finish", start_date, end_date)
    filtered = filter_df(wo_df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No historical work orders found for equipment_id '{equipment_id}'")

    filtered = filtered.sort_values("actual_finish").reset_index(drop=True)
    transition_matrix = get_transition_matrix(filtered, "primary_code")
    last_type = filtered.iloc[-1]["primary_code"]

    if last_type not in transition_matrix.index:
        return ErrorResult(error=f"No transition data for last work order type '{last_type}'")

    raw = sorted(transition_matrix.loc[last_type].items(), key=lambda t: t[1], reverse=True)

    predictions: List[NextWorkOrderEntry] = []
    for primary_code, prob in raw:
        entry = NextWorkOrderEntry(category="", primary_code=primary_code, primary_code_description="", probability=float(prob))
        if pfc_df is not None:
            match = pfc_df[pfc_df["primary_code"] == primary_code]
            if not match.empty:
                m = match.iloc[0]
                entry = NextWorkOrderEntry(
                    category=str(m.get("category", "")),
                    primary_code=primary_code,
                    primary_code_description=str(m.get("primary_code_description", "")),
                    probability=float(prob),
                )
        predictions.append(entry)

    return NextWorkOrderPredictionResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        last_work_order_type=last_type,
        predictions=predictions,
        message=f"Predicted next work order for '{equipment_id}' based on last type '{last_type}'.",
    )


def analyze_alert_to_failure(
    equipment_id: str,
    rule_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[AlertToFailureResult, ErrorResult]:
    """Analyze the relationship between a specific alert rule and subsequent maintenance events.

    Computes the probability that each alert occurrence leads to a work order (vs no
    maintenance) and the average time-to-maintenance in hours.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        rule_id: Alert rule identifier, e.g. ``"CR00002"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    alert_df = load("alert_events")
    if alert_df is None:
        return ErrorResult(error="Alert events data not available")

    try:
        parse_date(start_date)
        parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower(),
        "rule_id": lambda x, rid=rule_id: isinstance(x, str) and x.strip().lower() == rid.strip().lower(),
    }
    filtered = filter_df(alert_df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No alert events found for equipment '{equipment_id}' and rule '{rule_id}'")

    filtered = filtered.sort_values("start_time").reset_index(drop=True)

    transitions: List[str] = []
    time_diffs: List[float] = []
    for i in range(len(filtered) - 1):
        if str(filtered.iloc[i].get("rule_id", "")).strip().lower() == rule_id.strip().lower():
            for j in range(i + 1, len(filtered)):
                if str(filtered.iloc[j].get("event_group", "")).upper() == "WORK_ORDER":
                    transitions.append("WORK_ORDER")
                    diff = filtered.iloc[j]["start_time"] - filtered.iloc[i]["start_time"]
                    time_diffs.append(diff.total_seconds() / 3600)
                    break
            else:
                transitions.append("No Maintenance")

    if not transitions:
        return ErrorResult(error="Insufficient alert history to compute transitions")

    counts = Counter(transitions)
    total = len(transitions)

    entries: List[AlertToFailureEntry] = []
    for transition, count in sorted(counts.items(), key=lambda t: t[1], reverse=True):
        avg_hours = sum(time_diffs) / len(time_diffs) if transition == "WORK_ORDER" and time_diffs else None
        entries.append(
            AlertToFailureEntry(
                transition=transition,
                probability=count / total,
                average_hours_to_maintenance=avg_hours,
            )
        )

    return AlertToFailureResult(
        equipment_id=equipment_id,
        rule_id=rule_id,
        start_date=start_date,
        end_date=end_date,
        total_alerts_analyzed=total,
        transitions=entries,
        message=f"Analyzed {total} alert occurrences for rule '{rule_id}' on '{equipment_id}'.",
    )
