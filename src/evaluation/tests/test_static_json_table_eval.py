import pandas as pd

from evaluation.static_json_table_eval import evaluate_table


def test_static_json_table_eval_outputs_summary_and_details(tmp_path):
    input_path = tmp_path / "scenario_verification.csv"
    output_dir = tmp_path / "reports"

    df = pd.DataFrame(
        [
            {
                "id": 11,
                "Original question": "Break storage jobs into energy/material.",
                "final verified answer (ground truth)": "{'energy': 14, 'material': 48}",
                "Revised question": "Return JSON with energy and material.",
                "model_a": '{"energy": 14, "material": 27}',
                "model_b": '{"energy": 14, "material": 48}',
            },
            {
                "id": 12,
                "Original question": "Wear vs pressure.",
                "final verified answer (ground truth)": '{"winner": "wear", "wear": 75, "pressure": 9}',
                "Revised question": "Return JSON.",
                "model_a": '{"winner": "wear", "wear": 28, "pressure": 5}',
                "model_b": '{"winner": "wear", "wear": 75, "pressure": 9}',
            },
        ]
    )
    df.to_csv(input_path, index=False)

    paths = evaluate_table(
        input_path,
        id_column="id",
        gold_column="final verified answer (ground truth)",
        model_columns=["model_a", "model_b"],
        output_dir=output_dir,
    )

    details = pd.read_csv(paths["details"])
    summary = pd.read_csv(paths["summary"])

    assert len(details) == 4
    assert set(summary["model"]) == {"model_a", "model_b"}

    model_b = summary[summary["model"] == "model_b"].iloc[0]
    assert model_b["strict_accuracy"] == 1.0
    assert model_b["avg_f1"] == 1.0

    model_a = summary[summary["model"] == "model_a"].iloc[0]
    assert model_a["strict_accuracy"] == 0.0