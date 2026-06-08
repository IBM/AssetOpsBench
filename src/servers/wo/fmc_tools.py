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

from .data import load, write_failure_code
from .models import (
    ErrorResult,
    FmcCodeCount,
    FmcCodeDistributionResult,
    FmcWorkOrder,
    FmcWorkOrdersResult,
    FmcWriteResult,
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


def set_work_order_failure_code(wo_id: str, failure_code: str) -> Union[FmcWriteResult, ErrorResult]:
    """Write (impute) a failure code onto a failure-mode work order record.

    Persists ``failure_code`` to the ``wo_fmc`` record identified by ``wo_id``
    in CouchDB and returns the confirmed value.

    Args:
        wo_id: Work order identifier, e.g. ``"TST-WO00054"``.
        failure_code: Failure code to record, e.g. ``"Vibration"``.
    """
    code = (failure_code or "").strip()
    if not code:
        return ErrorResult(error="failure_code must be a non-empty string")
    result = write_failure_code(wo_id, code)
    if result is None:
        return ErrorResult(error="FMC work order data not available")
    if result is False:
        return ErrorResult(error=f"No work order found with wo_id '{wo_id}'")
    return FmcWriteResult(
        wo_id=wo_id,
        failure_code=code,
        updated=True,
        message=f"Recorded failure_code '{code}' on work order '{wo_id}'.",
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
