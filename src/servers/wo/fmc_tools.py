"""Failure-mode classification (FMC) tools for the Work Order MCP server.

These tools operate on the ``wo_fmc`` dataset — work orders keyed by ``wo_id``
(prefix ``TRN-`` for historical/training records, ``TST-`` for test records)
carrying a free-text ``description`` and a plain-text ``failure_code`` such as
``"Breakdown"`` or ``"Overheating"``.  This is distinct from the
equipment-keyed ``wo_events`` dataset used by the other work-order tools, which
classifies on structured ``MTxxx`` primary/secondary codes.

The workflow they support: read a work order, learn description-to-code
patterns from the historical (``train``) split, impute a failure code, write it
back, and rank failure codes by frequency.
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
_SPLIT_PREFIX = {"train": "TRN", "test": "TST"}
_VALID_SPLITS = ("all", "train", "test")


def _code(value) -> Optional[str]:
    """Normalise a ``failure_code`` cell to a non-empty string, or ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _apply_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter *df* to a ``wo_id`` prefix split (``train``/``test``); ``all`` is a no-op."""
    prefix = _SPLIT_PREFIX.get(split.lower())
    if prefix is None:
        return df
    return df[df["wo_id"].str.startswith(prefix)]


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


def list_work_order_failure_codes(split: str = "all") -> Union[FmcWorkOrdersResult, ErrorResult]:
    """List failure-mode work orders with their descriptions and recorded codes.

    Use ``split="train"`` for the historical/labelled records (to learn
    description-to-code patterns), ``split="test"`` for the records to be
    classified, or ``split="all"`` (default) for everything.

    Args:
        split: One of ``"all"``, ``"train"`` (TRN- records), or ``"test"`` (TST- records).
    """
    if split.lower() not in _VALID_SPLITS:
        return ErrorResult(error=f"split must be 'all', 'train', or 'test', got '{split}'")
    df = load(_FMC_DATASET)
    if df is None:
        return ErrorResult(error="FMC work order data not available")
    sub = _apply_split(df, split)
    if sub.empty:
        return ErrorResult(error=f"No work orders found for split '{split.lower()}'")

    items: List[FmcWorkOrder] = []
    labeled = 0
    for _, row in sub.iterrows():
        code = _code(row.get("failure_code"))
        if code is not None:
            labeled += 1
        items.append(
            FmcWorkOrder(
                wo_id=str(row["wo_id"]),
                description=str(row.get("description", "") or ""),
                failure_code=code,
            )
        )
    return FmcWorkOrdersResult(
        split=split.lower(),
        total=len(items),
        labeled=labeled,
        unlabeled=len(items) - labeled,
        work_orders=items,
        message=(
            f"Found {len(items)} work order(s) for split '{split.lower()}' "
            f"({labeled} labelled, {len(items) - labeled} unlabelled)."
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
    split: str = "all", top_n: Optional[int] = None
) -> Union[FmcCodeDistributionResult, ErrorResult]:
    """Rank failure codes by record count across the failure-mode dataset.

    Counts only records that have a recorded failure code, sorted by count
    descending.  Use ``split="train"`` to rank across historical records or
    ``split="test"`` to rank across the (imputed) test records.

    Args:
        split: One of ``"all"``, ``"train"``, or ``"test"``.
        top_n: If given, return only the top N codes.
    """
    if split.lower() not in _VALID_SPLITS:
        return ErrorResult(error=f"split must be 'all', 'train', or 'test', got '{split}'")
    df = load(_FMC_DATASET)
    if df is None:
        return ErrorResult(error="FMC work order data not available")
    sub = _apply_split(df, split)
    codes = [c for c in (_code(v) for v in sub.get("failure_code", [])) if c is not None]
    if not codes:
        return ErrorResult(error=f"No recorded failure codes for split '{split.lower()}'")

    counts = Counter(codes)
    ranked = counts.most_common(top_n)
    distribution = [FmcCodeCount(failure_code=code, count=count) for code, count in ranked]
    return FmcCodeDistributionResult(
        split=split.lower(),
        total_records=int(len(sub)),
        labeled_records=len(codes),
        distribution=distribution,
        message=(
            f"Ranked {len(distribution)} failure code(s) across {len(codes)} "
            f"labelled record(s) in split '{split.lower()}'."
        ),
    )
