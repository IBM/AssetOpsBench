"""Kaggle-style metric helpers for Industrial Automation Challenge.

Kaggle solution files use a column named ``usage`` to mark Public/Private
leaderboard rows. That column is not participant token usage and should not be
included in participant submissions. The participant-facing submission schema is:

    id,answer

This metric currently scores MCQA answer accuracy. Token efficiency, latency, and
reasoning completeness are described in the IJCAI proposal as final/offline audit
criteria; they need separate auditable fields or organizer-side logs before they
can be safely included in a live Kaggle score.
"""

from __future__ import annotations

import pandas as pd


ID_COLUMN = "id"
ANSWER_COLUMN = "answer"
USAGE_COLUMN = "usage"  # Kaggle Public/Private split marker, not token usage.


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = ID_COLUMN,
) -> float:
    """Return exact-match MCQA accuracy for Kaggle custom metric validation.

    Parameters follow Kaggle's common custom metric signature. Answers are
    compared case-insensitively after trimming whitespace.
    """

    row_id = row_id_column_name or ID_COLUMN
    _require_columns(solution, [row_id, ANSWER_COLUMN], "solution")
    _require_columns(submission, [row_id, ANSWER_COLUMN], "submission")

    merged = solution[[row_id, ANSWER_COLUMN]].merge(
        submission[[row_id, ANSWER_COLUMN]],
        on=row_id,
        how="left",
        suffixes=("_true", "_pred"),
    )
    if bool(merged[f"{ANSWER_COLUMN}_pred"].isna().any()):
        missing = merged.loc[merged[f"{ANSWER_COLUMN}_pred"].isna(), row_id].head(5).tolist()
        raise ValueError(f"Submission is missing predictions for row id(s): {missing}")

    y_true = merged[f"{ANSWER_COLUMN}_true"].map(_normalize_answer)
    y_pred = merged[f"{ANSWER_COLUMN}_pred"].map(_normalize_answer)
    return float((y_true == y_pred).mean())


def public_private_accuracy(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = ID_COLUMN,
) -> dict[str, float]:
    """Compute overall/Public/Private accuracy when a Kaggle usage split exists."""

    row_id = row_id_column_name or ID_COLUMN
    _require_columns(solution, [row_id, ANSWER_COLUMN], "solution")
    _require_columns(submission, [row_id, ANSWER_COLUMN], "submission")
    merged = solution.merge(
        submission[[row_id, ANSWER_COLUMN]],
        on=row_id,
        how="left",
        suffixes=("_true", "_pred"),
    )
    y_true = merged[f"{ANSWER_COLUMN}_true"].map(_normalize_answer)
    y_pred = merged[f"{ANSWER_COLUMN}_pred"].map(_normalize_answer)
    correct = y_true == y_pred
    out = {"overall": float(correct.mean())}
    if USAGE_COLUMN in merged.columns:
        for split_name in ("Public", "Private"):
            mask = merged[USAGE_COLUMN].astype(str).str.lower() == split_name.lower()
            if mask.any():
                out[split_name.lower()] = float(correct[mask].mean())
    return out


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _normalize_answer(value: object) -> str:
    return str(value).strip().upper()
