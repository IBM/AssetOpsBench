# AssetOpsBench Competition Starter Kit

This folder is the public submission kit for AssetOpsBench competitions. It is
designed for AssetOpsBench agent scenarios:

```text
competition/
├── run.py                       # command-line submission generator
├── eval_framework.py            # public packaging framework, no scoring
├── dataset_utils.py             # public dataset loader and ground-truth guard
├── metadata_config_phase1.json  # editable phase 1 config
├── metadata_config_phase2.json  # editable final phase 2 config
└── examples/baseline_predictor.py
```

## Public data rule

Do not publish or upload ground truth. Public Kaggle data should include only
fields such as:

```json
{"id": "301", "text": "What vibration analysis capabilities are available?", "type": "Vibration"}
```

The runner rejects records containing private evaluation fields such as
`expected_answer`, `correct_answer`, `answer`, `ground_truth`,
`characteristic_form`, or `scoring_method`.

## Quick start

Set `dataset.dataset_path` in `metadata_config_phase1.json` to the public
Kaggle JSONL file, and set `predictor.path` to the submission predictor:

```bash
python competition/run.py --config competition/metadata_config_phase1.json
```

The predictor is a `module:function` reference. The function receives an
`AssetOpsScenario` with `id`, `text`, and optional metadata. It can return a
string, or a dictionary:

```python
def predict(scenario):
    return {
        "prediction": "final answer text",
        "reasoning": "short optional rationale",
        "trajectory": [{"tool": "get_assets", "status": "ok"}],
    }
```

The generated zip contains:

- `submission.csv` with `id`, `prediction`, `reasoning`, and `trajectory`
- `meta_data.json` with model and track metadata

## Existing agent command

You can also wrap an existing CLI instead of a Python function:

```bash
python competition/run.py \
  --config competition/metadata_config_phase1.json \
  --agent-command 'uv run plan-execute --json {question_json}'
```

If the command prints JSON with an `answer` or `prediction` field, that value
is used. Otherwise stdout is used as the prediction.

## Final phase

One week before the competition ends, organizers should release the final phase
2 public dataset with the same public schema and no ground truth. Participants
should switch to `metadata_config_phase2.json`, generate the zip, and submit it
before the deadline. Final scoring can then be run offline by organizers against
the private ground-truth scenarios.
