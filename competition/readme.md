# Industrial Automation Challenge Competition Starter Kit

This folder contains the public AssetOpsBench starter kit for the IJCAI 2026
Industrial Automation Challenge. It is modeled after the CUREBench competition
workflow: load the public Kaggle scenarios, run a model or agent, generate a
valid submission file, and keep metadata for final/offline review.

Links:

- Website: https://sites.google.com/view/ai-industrial-challenge-ijcai/home
- Kaggle Track 1: https://www.kaggle.com/competitions/industrial-automation-challenge-track-1
- Kaggle Track 2: https://www.kaggle.com/competitions/industrial-automation-challenge-track-2
- Release branch: https://github.com/IBM/AssetOpsBench/tree/ijcai_2026_competition

## Current competition status

- Website changes are ready; publish after Kaggle setup is complete.
- Kaggle rules page is complete.
- Participant validation and test datasets have been uploaded.
- Ground-truth solution file has been uploaded with the Kaggle `usage` split
  column.
- Sample submission is expected to use the MCQA answer schema described below.
- Custom metric work is still in progress.

## Folder layout

```text
competition/
├── run.py                       # command-line submission generator
├── eval_framework.py            # public packaging framework, no hidden scoring
├── dataset_utils.py             # public dataset loader and leakage guard
├── metric.py                    # local/Kaggle-style exact-answer metric helper
├── metadata_config_phase1.json  # Track 1/internal-reasoning example config
├── metadata_config_phase2.json  # Track 2/agentic-reasoning example config
├── examples/
│   ├── baseline_predictor.py
│   ├── public_scenarios.jsonl
│   └── sample_submission.csv
└── tests/
```

## Tracks

### Track 1: Internal Model Reasoning

Models must reason using only internal parameters. External tools, APIs,
internet access, and retrieval mechanisms are not allowed during inference.
This track evaluates whether an open-weight model has internalized industrial
physics and Failure Mode and Effects Analysis (FMEA/FMECA) relationships.

Expected participant artifact:

- final MCQA answer letter
- reasoning trace / chain of thought for organizer-side review, where allowed by
  the submission channel
- open-source base model with no more than 8B parameters, per proposal rules

### Track 2: Agentic Tool-Augmented Reasoning

Models may operate as autonomous agents that invoke industrial tools during
reasoning. Potential tools include asset documentation, sensor telemetry APIs,
and historical maintenance/work-order logs.

Expected participant artifact:

- final MCQA answer letter
- reasoning trace
- tool-call trajectory/log for organizer-side review
- open-source base model or reproducible containerized solution, per proposal
  rules

## Public participant dataset schema

The challenge uses FailureSensorIQ-style multiple-choice industrial reasoning
questions. Public participant datasets should contain prompt information only;
hidden solution labels and leaderboard split markers must not be included in the
public test file.

Typical public row:

```json
{
  "id": "q123",
  "question_type": "open_ended_multi_choice",
  "passage": "Failure Mode and Effects Analysis (FMEA) maps failures to sensor variables...",
  "question": "Which sensor among the choices best correlates with loss of input power phase in an electric motor?",
  "options": {
    "A": "Phase current imbalance",
    "B": "Cooling water flow",
    "C": "Lubricant viscosity",
    "D": "Stack emission opacity"
  },
  "metadata": {
    "asset_class": "electric motor",
    "family": "positive_failure_to_sensor",
    "n_options": 4
  }
}
```

The loader also supports the older simple schema:

```json
{"id": "301", "text": "What vibration analysis capabilities are available?", "type": "Vibration"}
```

## Private solution schema

Organizer-side solution files currently use columns like:

```csv
id,anchor,answer,asset_class,family,n_options,usage
```

Important: Kaggle's `usage` column is the Public/Private leaderboard split
marker. It is not token usage. Valid values are typically `Public` and
`Private`. Do not include this column in participant datasets or participant
submissions.

If the live metric later incorporates token/cost efficiency, use a distinct
field name such as `token_usage`, `total_tokens`, `latency_ms`, or
`cost_estimate`, not `usage`, to avoid colliding with Kaggle's split marker.

## Public data leakage guard

`dataset_utils.py` rejects public scenario records containing private or
solution-like fields. These include:

- `answer`, `answers`, `correct_answer`, `expected_answer`
- `ground_truth`, `reference_answer`, `label`, `labels`, `target`
- `rubric`, `characteristic_form`, `scoring_method`
- `usage` because Kaggle uses it as the hidden Public/Private split marker

Validation data may include labels if organizers intentionally release a labeled
validation set. The final participant test set should not include answer labels.
Use `write_public_dataset(...)` to strip private fields from organizer-side JSON
or JSONL files before publishing a public artifact.

## Submission schema

For Kaggle MCQA scoring, the participant-facing submission should be:

```csv
id,answer
q123,A
```

Where:

- `id` matches the public test scenario id.
- `answer` is the selected option letter, e.g. `A`, `B`, `C`, ...

The starter kit can also carry `prediction`, `reasoning`, and `trajectory`
internally for offline packages, but the example Track 1 and Track 2 configs set
`submission_columns` to only `id,answer` so that generated `submission.csv` is
Kaggle-friendly.

## Quick start

From the repository root:

```bash
python competition/run.py --config competition/metadata_config_phase1.json
```

For Track 2 / agentic submissions:

```bash
python competition/run.py --config competition/metadata_config_phase2.json
```

The bundled baseline predictor always returns answer `A`; it exists only to
validate the submission pipeline. Replace
`competition/examples/baseline_predictor.py:predict` with your own predictor in
the config.

## Predictor interface

The predictor is a `module:function` reference. The function receives an
`AssetOpsScenario` with:

- `id`
- `text` — composed prompt text from passage/question/options
- `metadata` — asset class, family, number of options, and other safe metadata

Example:

```python
def predict(scenario):
    return {
        "answer": "A",
        "prediction": "A",
        "reasoning": "short optional rationale",
        "trajectory": [{"tool": "get_sensors", "status": "ok"}],
    }
```

You can also wrap an existing CLI instead of a Python function:

```bash
python competition/run.py \
  --config competition/metadata_config_phase2.json \
  --agent-command 'uv run plan-execute --json {question_json}'
```

If the command prints JSON with `answer`, `choice`, or `prediction`, that value
is used. Otherwise stdout becomes the answer/prediction text.

## Generated files

The runner writes to `competition_results/` by default:

- `submission.csv` — by default `id,answer` for the Kaggle configs
- `meta_data.json` — model/track metadata for organizer review
- `submission.zip` — CSV + metadata package for offline/CUREBench-style review

If Kaggle requires direct CSV upload, upload `submission.csv`. If organizers ask
for a package, upload `submission.zip`.

## Metadata

Required metadata fields mirror the CUREBench-style package:

- `model_name`
- `track`
- `base_model_type`
- `base_model_name`
- `dataset`

Current allowed examples:

- `track`: `internal_reasoning`, `agentic_reasoning`
- `base_model_type`: `API`, `OpenWeighted`, `Hybrid`

For final/offline review, participants should also document:

- model size and whether it satisfies the ≤8B parameter constraint
- prompting strategy, e.g. CoT/ReAct/planner-executor
- whether external tools were used
- average token usage, latency, and tool-call counts if available
- reproducibility details: model weights or containerized solution

## Metric status

`competition/metric.py` provides a local/Kaggle-style exact MCQA answer accuracy
helper using the common Kaggle signature:

```python
score(solution, submission, row_id_column_name="id")
```

It expects:

- solution columns: `id`, `answer` plus optional Kaggle split column `usage`
- submission columns: `id`, `answer`

The IJCAI proposal mentions a broader final rubric with MCQ accuracy, latency,
reasoning completeness, and token efficiency. Live token-efficiency scoring
requires an auditable token/cost field or organizer-side execution logs. Until
that field is finalized, the safe Kaggle live metric is answer accuracy, with
latency/reasoning/token efficiency handled in final/offline review.

## Local checks

```bash
python -m pytest competition/tests -q
python competition/run.py --config competition/metadata_config_phase1.json --subset-size 1
```

Expected smoke-test output:

```text
Processed scenarios: 1
Submission package: competition_results/submission.zip
```

## Organizer notes

The local parent folder currently contains organizer-side artifacts, including:

- `iso_sensors_mcqa_val.jsonl`
- `HIDDEN_iso_sensors_mcqa_test_ground_truth_with_usage.jsonl`
- `HIDDEN_iso_sensors_mcqa_test_ground_truth_with_usage.CSV`
- `IJCAI Data Management.ipynb`

Do not commit real hidden solution files to the public competition branch. Only
commit toy/example rows and code needed to regenerate public-safe files.
