"""Failure-mode classification (FMC) tools for the Work Order MCP server.

These tools operate on the ``wo_fmc`` dataset — work orders keyed by ``wo_id``
(prefix ``TRN-`` for historical/training records, ``TST-`` for test records)
carrying a free-text ``description`` and a plain-text ``failure_code`` such as
``"Breakdown"`` or ``"Overheating"``.  This is distinct from the
equipment-keyed ``wo_events`` dataset used by the other work-order tools, which
classifies on structured ``MTxxx`` primary/secondary codes.

The workflow they support: read a work order, learn description-to-code
patterns from the already-labelled records, impute a failure code, write it
back, and rank failure codes by frequency.

Records are filtered by whether a failure code has been recorded
(``labeled``) rather than by any train/test tag — the ``wo_fmc`` dataset
carries no such tag.
"""

from collections import Counter
from typing import List, Optional, Union

import pandas as pd

from .data import load, write_failure_codes
from .models import (
    ErrorResult,
    FmcBatchWriteResult,
    FmcCodeAssignment,
    FmcCodeCount,
    FmcCodeDistributionResult,
    FmcWorkOrder,
    FmcWorkOrdersResult,
    FmcWriteEntry,
)

_FMC_DATASET = "wo_fmc"


def _code(value) -> Optional[str]:
    """Normalise a ``failure_code`` cell to a non-empty string, or ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _is_labeled(value) -> bool:
    """True when the record has a recorded (non-blank) failure code."""
    return _code(value) is not None


def get_work_order_failure_code(wo_id: str) -> Union[FmcWorkOrder, ErrorResult]:
    """Retrieve a single failure-mode work order by its ``wo_id``.

    Returns the work order's free-text description and its recorded failure
    code.  ``failure_code`` is null when none has been recorded yet.

    Args:
        wo_id: Work order identifier, e.g. ``"TST-WO00032"``.
    """
    df = load(_FMC_DATASET)
    if df is None:
        return ErrorResult(error="FMC work order data not available")
    match = df[df["wo_id"] == wo_id]
    if match.empty:
        return ErrorResult(error=f"No work order found with wo_id '{wo_id}'")
    row = match.iloc[0]
    return FmcWorkOrder(
        wo_id=str(row["wo_id"]),
        description=str(row.get("description", "") or ""),
        failure_code=_code(row.get("failure_code")),
    )


def list_work_order_failure_codes(
    labeled: Optional[bool] = None,
) -> Union[FmcWorkOrdersResult, ErrorResult]:
    """List failure-mode work orders with their descriptions and recorded codes.

    Use ``labeled=True`` for the records that already have a failure code (to
    learn description-to-code patterns), ``labeled=False`` for the blank
    records still to be classified, or omit it (default) for everything.

    Args:
        labeled: If ``True``, only records with a recorded failure code; if
            ``False``, only records with a blank failure code; if omitted, all.
    """
    df = load(_FMC_DATASET)
    if df is None:
        return ErrorResult(error="FMC work order data not available")

    items: List[FmcWorkOrder] = []
    labeled_count = 0
    for _, row in df.iterrows():
        code = _code(row.get("failure_code"))
        if labeled is True and code is None:
            continue
        if labeled is False and code is not None:
            continue
        if code is not None:
            labeled_count += 1
        items.append(
            FmcWorkOrder(
                wo_id=str(row["wo_id"]),
                description=str(row.get("description", "") or ""),
                failure_code=code,
            )
        )
    if not items:
        return ErrorResult(error="No matching work orders found")

    return FmcWorkOrdersResult(
        total=len(items),
        labeled=labeled_count,
        unlabeled=len(items) - labeled_count,
        work_orders=items,
        message=(
            f"Found {len(items)} work order(s) "
            f"({labeled_count} labelled, {len(items) - labeled_count} unlabelled)."
        ),
    )


def set_work_order_failure_codes(
    assignments: List[FmcCodeAssignment],
) -> Union[FmcBatchWriteResult, ErrorResult]:
    """Write (impute) failure codes onto one or more work order records.

    Persists each ``failure_code`` to its ``wo_fmc`` record in CouchDB in a
    single batch.  Pass a one-element list to set a single record, or many to
    impute a whole batch at once (preferred over many single calls).

    Args:
        assignments: List of ``{"wo_id": ..., "failure_code": ...}`` items,
            e.g. ``[{"wo_id": "TST-WO00054", "failure_code": "Vibration"}]``.
    """
    if not assignments:
        return ErrorResult(error="assignments must be a non-empty list")

    updates: dict = {}
    for a in assignments:
        code = (a.failure_code or "").strip()
        if not code:
            return ErrorResult(error=f"failure_code for '{a.wo_id}' must be a non-empty string")
        if a.wo_id in updates:
            return ErrorResult(error=f"duplicate wo_id in assignments: '{a.wo_id}'")
        updates[a.wo_id] = code

    status = write_failure_codes(updates)
    if status is None:
        return ErrorResult(error="FMC work order data not available")

    results = [
        FmcWriteEntry(wo_id=wo_id, failure_code=code, updated=bool(status.get(wo_id)))
        for wo_id, code in updates.items()
    ]
    updated = sum(1 for r in results if r.updated)
    missing = [r.wo_id for r in results if not r.updated]
    message = f"Recorded {updated}/{len(results)} failure code(s)."
    if missing:
        message += f" No record found for: {', '.join(missing)}."
    return FmcBatchWriteResult(
        total=len(results),
        updated=updated,
        results=results,
        message=message,
    )


def get_failure_code_distribution(
    top_n: Optional[int] = None,
) -> Union[FmcCodeDistributionResult, ErrorResult]:
    """Rank recorded failure codes by record count across the failure-mode dataset.

    Counts every record that has a recorded failure code, sorted by count
    descending.  (Blank records are ignored, so this ranks the labelled
    population.)

    Args:
        top_n: If given, return only the top N codes.
    """
    df = load(_FMC_DATASET)
    if df is None:
        return ErrorResult(error="FMC work order data not available")
    codes = [c for c in (_code(v) for v in df.get("failure_code", [])) if c is not None]
    if not codes:
        return ErrorResult(error="No recorded failure codes found")

    counts = Counter(codes)
    ranked = counts.most_common(top_n)
    distribution = [FmcCodeCount(failure_code=code, count=count) for code, count in ranked]
    return FmcCodeDistributionResult(
        total_records=int(len(df)),
        labeled_records=len(codes),
        distribution=distribution,
        message=(
            f"Ranked {len(distribution)} failure code(s) across "
            f"{len(codes)} labelled record(s)."
        ),
    )
