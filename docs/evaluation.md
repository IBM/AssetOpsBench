# Evaluation

Offline scoring of saved agent trajectories against ground-truth scenarios.

The evaluation module follows the three-stage pattern used by SWE-bench,
HELM, and τ-bench:

```
agent run  →  trajectory (run_id)  →  evaluate  →  reports/<run_id>.json
```

Re-scoring from saved trajectories is first-class: re-run with a
different scorer or judge model without re-invoking the agent.

## Concepts

The vocabulary follows MLflow's evaluation split:

- **Scenario** — a ground-truth record on disk. Carries `id`, `text`
  (the utterance), `type`, `characteristic_form` (expected behaviour),
  and optional `scoring_method`.
- **Trajectory** — a per-run JSON file persisted by the agent runners
  when `AGENT_TRAJECTORY_DIR` is set. Carries `run_id`, `scenario_id`,
  `question`, `answer`, and per-turn detail.
- **Scorer** — a callable that takes
  `(scenario, answer, trajectory_text)` and returns a `ScorerResult`.
  Scorers fall into three families:
  - **Code-Based** — deterministic, no LLM (e.g. `exact_string_match`,
    `numeric_match`). *Skeleton only* in this branch.
  - **LLM-As-Judge** — `llm_judge`. Six-criterion rubric, requires a
    LiteLLM-routable model passed via `--judge-model`.
  - **Semantic-Score** — similarity-based, no LLM call. *Skeleton only*
    in this branch.
- **Evaluator** — orchestrates a batch: loads scenarios + trajectories,
  joins on `scenario_id`, dispatches to scorers, aggregates results.

## Inputs

### Scenario file

JSON list, JSON object, or JSONL. Fields the scorer cares about:

| Field                 | Used by                                          | Notes                                          |
| --------------------- | ------------------------------------------------ | ---------------------------------------------- |
| `id`                  | join                                             | Coerced to string at load time                 |
| `text`                | all                                              | The utterance the agent answered               |
| `type`                | reporting                                        | Scenario family (`iot`, `tsfm`, `FMSR`, …)     |
| `characteristic_form` | `llm_judge`, `semantic_similarity`*              | Expected behaviour, free-form                  |
| `expected_answer`     | `exact_string_match`*, `numeric_match`*          | Exact target string / number                   |
| `scoring_method`      | dispatch                                         | Registered scorer name; overrides CLI default  |
| `tolerance`           | `numeric_match`*                                 | Optional relative + absolute tolerance         |

\* Skeleton in this branch — see [Available scorers](#available-scorers-in-this-branch).

Ground-truth files under `groundtruth/` already match this schema —
they're a drop-in scenarios input.

### Trajectory file

Written by the observability layer to `AGENT_TRAJECTORY_DIR` as one
JSON per run. Fields the evaluator reads:

```
{
  "run_id":      "<uuid>",
  "scenario_id": "<scenario id>",
  "runner":      "claude-agent" | "plan-execute" | …,
  "model":       "<provider/model>",
  "question":    "<utterance>",
  "answer":      "<final response>",
  "trajectory":  {…}  // SDK Trajectory dict, or list[StepResult] for plan-execute
}
```

`scenario_id` is critical — trajectories with `null` scenario_id are
dropped at the join step. Pass `--scenario-id` to the agent CLI to set it.

## End-to-end workflow

```bash
# 1. Persist trajectories under AGENT_TRAJECTORY_DIR
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories
uv run claude-agent "List all failure modes of asset Chiller." --scenario-id 101

# 2. Score with LLM-As-Judge against the ground-truth file
uv run evaluate \
  --trajectories traces/trajectories \
  --scenarios groundtruth/101.json \
  --scorer-default llm_judge \
  --judge-model litellm_proxy/aws/claude-opus-4-6
```

Output:

```
Scenarios: 1  Passed: 1  Pass rate: 100.0%

By scenario type:
  FMSR             1/1   (100.0%)

Operational metrics:
  tokens_in_total:   7
  tokens_out_total:  25
  tool_calls_total:  1
  duration_ms_p50:   14690.6

Reports written: reports/<run_id>.json (1 files)
Aggregate:       reports/_aggregate.json
```

## Output layout

```
reports/
├── <run_id>.json        # one ScenarioResult per trajectory
├── <run_id>.json
└── _aggregate.json      # EvalReport: totals, by_scenario_type, ops rollup
```

Per-run file (`reports/<run_id>.json`):

```json
{
  "scenario_id": "101",
  "scenario_type": "FMSR",
  "run_id": "112c1b56-…",
  "runner": "claude-agent",
  "model": "litellm_proxy/aws/claude-opus-4-6",
  "question": "List all failure modes of asset Chiller.",
  "answer":   "Here are the 7 failure modes for the Chiller asset: …",
  "score": {
    "scorer": "llm_judge",
    "passed": true,
    "score": 1.0,
    "rationale": "",
    "details": {
      "task_completion": true,
      "data_retrieval_accuracy": true,
      "generalized_result_verification": true,
      "agent_sequence_correct": true,
      "clarity_and_justification": true,
      "hallucinations": false,
      "suggestions": ""
    }
  },
  "ops": {
    "turn_count": 2,
    "tool_call_count": 1,
    "unique_tools": ["get_failure_modes"],
    "tokens_in": 7,
    "tokens_out": 25,
    "duration_ms": 14690.6,
    "est_cost_usd": 0.001959
  }
}
```

Aggregate (`reports/_aggregate.json`) is the full `EvalReport`:

```json
{
  "generated_at": "<iso8601>",
  "runners": ["claude-agent"],
  "models":  ["litellm_proxy/aws/claude-opus-4-6"],
  "totals": {
    "scenarios": 1,
    "scored":    1,
    "passed":    1,
    "pass_rate": 1.0
  },
  "by_scenario_type": {
    "FMSR": {"total": 1, "passed": 1, "pass_rate": 1.0}
  },
  "ops": {
    "tokens_in_total":    7,
    "tokens_out_total":   25,
    "tool_calls_total":   1,
    "duration_ms_p50":    14690.6,
    "duration_ms_p95":    14690.6,
    "est_cost_usd_total": 0.001959
  },
  "results": [ /* one ScenarioResult per run, same shape as the per-run files */ ]
}
```

## CLI reference

```
uv run evaluate \
  --trajectories DIR_OR_FILE     # required
  --scenarios FILE [FILE ...]    # required, one or more
  [--reports-dir DIR]            # default: reports/
  [--scorer-default NAME]        # default: llm_judge
  [--judge-model MODEL_ID]       # required when llm_judge runs
  [-v]
```


## Available scorers in this branch

| Family        | Registered name        | Status                                      |
| ------------- | ---------------------- | ------------------------------------------- |
| LLM-As-Judge  | `llm_judge`            | Works. Installed by passing `--judge-model` |
| Code-Based    | `exact_string_match`   | **Skeleton — `NotImplementedError`**        |
| Code-Based    | `numeric_match`        | **Skeleton — `NotImplementedError`**        |
| Semantic-Score| `semantic_similarity`  | **Skeleton — `NotImplementedError`**        |

Skeleton scorers don't auto-register; calling them raises
`NotImplementedError`. Fill in the body and call
`evaluation.scorers.register("<name>", <fn>)` to enable.

## LLM-As-Judge

Six-criterion rubric, prompt mirrored from
`src/tmp/evaluation_agent/result_evaluation_prompt.py`:

- `task_completion`
- `data_retrieval_accuracy`
- `generalized_result_verification`
- `agent_sequence_correct`
- `clarity_and_justification`
- `hallucinations`

A run passes overall iff the first five are `true` **and**
`hallucinations` is `false`. The score is the fraction of the first
five satisfied, minus 0.2 if `hallucinations` is `true`. The judge's
free-form `suggestions` (or legacy `reason`) lands in
`score.rationale`; the full review dict lands in `score.details`.

To customise: edit `_PROMPT_TEMPLATE` in
`src/evaluation/scorers/llm_judge.py`.

## Programmatic use

```python
from pathlib import Path
from evaluation import Evaluator
from evaluation.scorers.llm_judge import install
from llm import LiteLLMBackend

install(LiteLLMBackend(model_id="litellm_proxy/aws/claude-opus-4-6"))

report = Evaluator(default_scorer="llm_judge").evaluate(
    trajectories_path=Path("traces/trajectories"),
    scenarios_paths=[Path("groundtruth/101.json")],
)

for r in report.results:
    print(r.run_id, r.score.passed, r.score.score)
```

## Plug in a custom scorer

```python
from evaluation import scorers
from evaluation.models import ScorerResult

def keyword_hit(scenario, answer, trajectory_text) -> ScorerResult:
    required = (scenario.model_extra or {}).get("required_keywords", [])
    hits = [k for k in required if k.lower() in answer.lower()]
    passed = len(hits) == len(required)
    return ScorerResult(
        scorer="keyword_hit",
        passed=passed,
        score=len(hits) / max(1, len(required)),
        rationale="" if passed else f"missing: {set(required) - set(hits)}",
    )

scorers.register("keyword_hit", keyword_hit)
# Any scenario with "scoring_method": "keyword_hit" now routes here.
```

## Loop over all ground-truth files

```bash
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories

for f in groundtruth/*.json; do
  utt=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['text'])" "$f")
  sid=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['id'])"  "$f")
  uv run claude-agent "$utt" --scenario-id "$sid"
done

uv run evaluate \
  --trajectories traces/trajectories \
  --scenarios groundtruth/*.json \
  --scorer-default llm_judge \
  --judge-model litellm_proxy/aws/claude-opus-4-6
```
