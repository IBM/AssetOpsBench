"""Evaluate structured model outputs from a scenario verification CSV.

Expected CSV shape:

id, Original question, final verified answer (ground truth), Revised question, model_1, model_2, ...

The script compares the ground-truth column against each model-output column
and writes:

1. A row-level detailed CSV.
2. A model-level summary CSV.

Example:
    uv run python -m evaluation.static_json_table_eval \
      --input scenario_verification.csv \
      --gold-column "final verified answer (ground truth)" \
      --id-column id \
      --output-dir reports/static_json_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.static_json_score import evaluate_static_json


DEFAULT_NON_MODEL_COLUMNS = {
    "id",
    "Original question",
    "original question",
    "final verified answer (ground truth)",
    "Revised question",
    "revised question",
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def infer_model_columns(
    columns: list[str],
    *,
    id_column: str,
    gold_column: str,
    question_columns: list[str],
) -> list[str]:
    """Infer which columns contain model outputs."""
    excluded = set(DEFAULT_NON_MODEL_COLUMNS)
    excluded.add(id_column)
    excluded.add(gold_column)
    excluded.update(question_columns)

    return [column for column in columns if column not in excluded]


def evaluate_table(
    input_path: str | Path,
    *,
    id_column: str,
    gold_column: str,
    model_columns: list[str] | None = None,
    question_columns: list[str] | None = None,
    output_dir: str | Path = "reports/static_json_eval",
) -> dict[str, Path]:
    """Evaluate all model columns in a scenario verification CSV."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_columns = question_columns or ["Original question", "Revised question"]

    df = pd.read_csv(input_path)

    if id_column not in df.columns:
        raise ValueError(f"Missing id column: {id_column}")

    if gold_column not in df.columns:
        raise ValueError(f"Missing gold column: {gold_column}")

    if model_columns is None:
        model_columns = infer_model_columns(
            list(df.columns),
            id_column=id_column,
            gold_column=gold_column,
            question_columns=question_columns,
        )

    detailed_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        scenario_id = row[id_column]
        gold_answer = row[gold_column]

        if pd.isna(gold_answer):
            continue

        for model_column in model_columns:
            model_answer = row.get(model_column)

            if pd.isna(model_answer):
                continue

            score = evaluate_static_json(gold_answer, model_answer)

            detailed_rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model_column,
                    "strict_match": score.strict_exact_match_accuracy,
                    "partial_exact": score.partial_exact_match_accuracy,
                    "partial_similarity": score.partial_similarity_score,
                    "precision": score.precision,
                    "recall": score.recall,
                    "f1": score.f1,
                    "total_gold_keys": score.total_gold_keys,
                    "total_model_keys": score.total_model_keys,
                    "matched_keys": score.matched_keys,
                    "exact_value_matches": score.exact_value_matches,
                    "missing_keys": _json_dumps(score.missing_keys),
                    "extra_keys": _json_dumps(score.extra_keys),
                    "details": _json_dumps(
                        [detail.__dict__ for detail in score.details]
                    ),
                }
            )

    detailed_df = pd.DataFrame(detailed_rows)

    detailed_path = output_dir / "static_json_eval_details.csv"
    summary_path = output_dir / "static_json_eval_summary.csv"

    detailed_df.to_csv(detailed_path, index=False)

    if detailed_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "model",
                "scenarios",
                "strict_accuracy",
                "avg_partial_exact",
                "avg_partial_similarity",
                "avg_precision",
                "avg_recall",
                "avg_f1",
            ]
        )
    else:
        summary_df = (
            detailed_df.groupby("model")
            .agg(
                scenarios=("scenario_id", "count"),
                strict_accuracy=("strict_match", "mean"),
                avg_partial_exact=("partial_exact", "mean"),
                avg_partial_similarity=("partial_similarity", "mean"),
                avg_precision=("precision", "mean"),
                avg_recall=("recall", "mean"),
                avg_f1=("f1", "mean"),
            )
            .reset_index()
        )

    summary_df.to_csv(summary_path, index=False)

    return {
        "details": detailed_path,
        "summary": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate structured model outputs from a scenario CSV."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--id-column",
        default="id",
        help="Scenario ID column name",
    )
    parser.add_argument(
        "--gold-column",
        default="final verified answer (ground truth)",
        help="Ground-truth answer column name",
    )
    parser.add_argument(
        "--model-column",
        action="append",
        default=None,
        help="Model output column to evaluate. Can be repeated. "
        "If omitted, model columns are inferred.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/static_json_eval",
        help="Directory for output CSV reports",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = evaluate_table(
        args.input,
        id_column=args.id_column,
        gold_column=args.gold_column,
        model_columns=args.model_column,
        output_dir=args.output_dir,
    )

    print(f"Wrote details: {paths['details']}")
    print(f"Wrote summary: {paths['summary']}")


if __name__ == "__main__":
    main()