"""Kaggle-style metric helper for Industrial Automation Challenge.

The participant-facing submission schema is:

    id,answer

This helper scores exact MCQA answer accuracy. Broader criteria such as token
efficiency, latency, and reasoning completeness require an announced/auditable
field or organizer-side logs before they should be included in live scoring.
"""

from __future__ import annotations

import pandas as pd


ID_COLUMN = "id"
ANSWER_COLUMN = "answer"


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = ID_COLUMN,
) -> float:
    """Return exact-match MCQA accuracy for Kaggle custom metric validation.

    Parameters follow Kaggle's common custom metric signature. Answers are
    compared case-insensitively after trimming whitespace. The public repository
    contains only toy/example data; real evaluation data should remain outside
    the public branch.
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


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def _normalize_answer(value: object) -> str:
    return str(value).strip().upper()
