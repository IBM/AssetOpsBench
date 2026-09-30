# Run agents and evaluate their results

| Step | What happens | Output |
|---|---|---|
| **Execute** | An agent completes each task using the benchmark MCP tools. | Answer, trajectory, trace and execution metrics |
| **Grade** | An independent judge scores the saved answer and trajectory against the scenario rubric. | Pass/fail, score, rubric results and rationale |

**Choose:** [grade existing results](#grade-existing-results) · [execute one model](#execute-one-model) · [compare five models](#compare-five-models) · [repeat three times](#repeat-three-times)

All commands run from the repository root. The committed Transformer suite contains **50 open-form tasks and 2 negative scenarios**. No scenario generation is needed.

## Setup

**Install dependencies** — requires Python 3.12+ and `uv`.

```bash
uv sync
```

**Authenticate the Claude executor/judge** — use an installed Claude Code CLI that supports `--safe-mode`.

```bash
claude auth login
claude auth status
claude --help
```

For grading saved results, continue directly to [grade existing results](#grade-existing-results). Execution also requires the database and model access below.

**Create local configuration** — preserve an existing `.env`.

```bash
test -f .env || cp .env.public .env
```

Edit `.env` with the following settings. Keep credentials out of Git.

| Setting | Required for |
|---|---|
| `COUCHDB_URL=http://localhost:5984` | Local database |
| `COUCHDB_USERNAME=admin`, `COUCHDB_PASSWORD=password` | Credentials matching the supplied Docker Compose service |
| `ZAI_API_KEY` | GLM execution; optional `ZAI_BASE_URL` selects its API endpoint |
| `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` | FMSR tools that invoke their own Watsonx model |

**Start the database and load default fixtures** — requires Docker; use a dedicated benchmark instance. Container startup reloads its default data.

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
docker compose -f src/couchdb/docker-compose.yaml logs -f couchdb
```

Wait for `All databases initialised`, then press Ctrl-C to leave the log viewer. MCP servers start automatically during execution.

### Execution models

| Target name | `--agent` | `--model-id` | Access |
|---|---|---|---|
| `opus-5-5` | `claude` | `claude-opus-5-5` | Claude Code authentication |
| `gpt-6-astra` | `codex` | `gpt-6-astra` | Authenticated Codex CLI |
| `glm-5-3-low` | `openai` | `zai/glm-5.3` | `ZAI_API_KEY`; add `--reasoning-effort low` |
| `gpt-6-1-sol` | `codex` | `gpt-6.1-sol` | Authenticated Codex CLI |
| `fable-5-1` | `claude` | `claude-fable-5-1` | Claude Code authentication |

Exact model IDs must be available to your account. Codex execution uses subscription authentication rather than `OPENAI_API_KEY`.

**Authenticate Codex** — needed only for the Codex targets.

```bash
codex login
codex --version
```

The single-model runner uses `codex` on PATH; `BENCHMARK_CODEX_EXECUTABLE` can select another installed executable. The five-model launcher currently overrides that setting with this macOS app path:

```text
/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex
```

On another installation, update that executable assignment in [`tools/run_generated_comparison.py`](../tools/run_generated_comparison.py) before using the launcher. Its CLI must support the flags used by [`CodexAgentRunner`](../src/agent/codex_agent/runner.py).

## Grade existing results

**Regrade the committed Opus trajectories with Fable 5.1.** Requires Claude Code authentication; no database or agent execution is needed. This calls the judge and writes a new report.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --trajectories benchmarks/runs/2026-09-30-transformer/models/opus-5-5/trajectories \
  --scenarios benchmarks/runs/2026-09-30-transformer/suite/scenarios.json \
              benchmarks/runs/2026-09-30-transformer/suite/negative_scenarios.json \
  --judge-model claude-code/claude-fable-5-1 \
  --reports-dir generated/reports/transformer-opus-fable
```

Read `generated/reports/transformer-opus-fable/_aggregate.json`. Its `results` array contains each scenario's score, rubric booleans and rationale. To grade another model, replace `opus-5-5` with its target name. For `fable-5-1`, also add `--allow-self-judge`; the judge still starts in a separate, tool-free session.

`--trajectories` also accepts one trajectory JSON file. Supply the matching scenario files, not the measurement or raw trace directories. Regrading here does not update the published measurements or comparison HTML.

## Execute one model

This runner uses your configured database directly. Tasks can create or change records; it does **not** clone or reset the database. Use the [five-model launcher](#compare-five-models) for isolation between models.

**Preview one task** — validates the suite and prints its selection without executing an agent or judge.

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.generated_suite_runner \
  benchmarks/runs/2026-09-30-transformer/suite \
  --output-dir generated/comparisons/transformer-single \
  --name opus-5-5 --agent claude --model-id claude-opus-5-5 \
  --judge-model claude-code/claude-fable-5-1 \
  --limit 1 --dry-run
```

**Execute and grade one task** — the agent calls tools, then a separate Fable session grades its saved trajectory.

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.generated_suite_runner \
  benchmarks/runs/2026-09-30-transformer/suite \
  --output-dir generated/comparisons/transformer-single \
  --name opus-5-5 --agent claude --model-id claude-opus-5-5 \
  --judge-model claude-code/claude-fable-5-1 \
  --limit 1
```

**Execute the full suite** — rerun the command above without `--limit 1`. Matching completed trajectories are reused; the remaining tasks execute in suite order. Grading runs after the target's execution pass. Use a new output directory for a fresh repetition or changed settings.

| Flag | Effect |
|---|---|
| `--name`, `--agent`, `--model-id` | Select a target using the model table above |
| Omit `--judge-model` | Execute only; grade saved trajectories later |
| `--allow-self-judge` | Permit Fable execution plus independent Fable judging |
| `--timeout 900` | Whole agent invocation timeout in seconds; default 900 |
| `--max-invocation-attempts 3` | Maximum retained whole-invocation attempts per task; default 3 |
| `--reasoning-effort low` | Explicit GLM reasoning; options are `low`, `high`, `max` |

The runner records unknown effective token limits, temperature and reasoning defaults as `null`. The scenario generator's 128k budget is not an execution token cap.

## Compare five models

**Execute all five models and grade as each task finishes.** Requires all model authentication and a populated database. Use a fresh output directory.

```bash
PYTHONPATH=src .venv/bin/python tools/run_generated_comparison.py \
  --output-dir generated/comparisons/transformer-new
```

Configuration: [`benchmarks/generated-comparison.json`](../benchmarks/generated-comparison.json). The launcher captures one database snapshot, clones an isolated namespace for each model, and runs five execution targets concurrently. Each target processes scenarios serially; database state persists within that target, including writes from retries. There is no per-scenario reset.

Each target has a separate grading worker. Every graded scenario starts a fresh, tool-free Fable 5.1 session, including Fable's own execution results. Execution and grading durations are measured separately. The launcher writes `comparison.html` when finished.

## Inspect progress and results

**Open the live dashboard in a second terminal** — reads all model folders beside the selected target.

```bash
PYTHONPATH=src .venv/bin/python tools/live_evaluation/server.py \
  --target generated/comparisons/transformer-new/opus-5-5 \
  --port 8770
```

Open [http://127.0.0.1:8770](http://127.0.0.1:8770). For a single-model run, replace `transformer-new` with `transformer-single`.

**Follow execution and grading logs** — change the target name for another model. Later retries use `suite-attempt-2.log`, etc.

```bash
tail -f generated/comparisons/transformer-new/opus-5-5/suite-attempt-1.log
```

```bash
tail -f generated/comparisons/transformer-new/opus-5-5/live-grading.log
```

**Build or refresh the HTML report** — reads measured results; invokes no model.

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.comparison_report \
  generated/comparisons/transformer-new \
  --output generated/comparisons/transformer-new/comparison.html
```

Open the HTML locally. GitHub's source view does not render it.

### Saved files

Each model folder contains:

| Path | Contents |
|---|---|
| `settings.json` | Exact model/harness/runtime versions, suite/rubric hashes, limits and policies |
| `environment.json` | Snapshot hash and database isolation/reset policy; created by the comparison launcher |
| `measurements/*.json` | Every invocation attempt: status, start/end, duration, metrics, errors and separately timed grading |
| `trajectories/*.json` | Saved answers and structured trajectories used by the evaluator |
| `traces/*.jsonl` | Full observed messages, tool arguments/outputs/errors and timestamps; separate judge traces |
| `reports/_aggregate.json` | Graded results, rubric details, rationale and totals |
| `database_audit/*.jsonl` | Database writes and record changes captured by the comparison launcher |

Missing measurements stay `null`. CLI cost estimates are separate from actual billed cost. First-response latency is the first observed response event, not streaming first-token latency. Whole-invocation timing covers startup, tools, persistence and process exit.

## Repeat three times

**Keep a completed comparison as repetition 1 and execute two more.** This checks that the suite, models, judge and starting database snapshot match. The published snapshot itself is not bundled; default Docker fixtures may not match it. To repeat your own fresh comparison, use its model-folder root as `--baseline` and restore its original starting database state first.

```bash
PYTHONPATH=src .venv/bin/python tools/run_repeated_comparison.py \
  --baseline benchmarks/runs/2026-09-30-transformer/models \
  --output-dir generated/comparisons/transformer-k3-new
```

Repetitions run serially; the five models and grading workers run concurrently within each repetition.

**Watch the repeated comparison.**

```bash
PYTHONPATH=src .venv/bin/python tools/live_evaluation/server.py \
  --experiment generated/comparisons/transformer-k3-new/experiment.json \
  --port 8771
```

Open [http://127.0.0.1:8771](http://127.0.0.1:8771).

**Publish graphs, README, CSV/JSON and HTML after all three repetitions finish.** Writes the chosen local artifact folder; does not commit or push.

```bash
uv run tools/publish_repeated_comparison.py \
  --experiment generated/comparisons/transformer-k3-new/experiment.json \
  --output-dir benchmarks/runs/transformer-k3-new
```

Results show equal-weight means and sample standard deviations across complete repetitions, pooled timing/resource metrics and per-scenario pass frequency. Retry attempts are not additional repetitions; passing once out of three is not reported as the average pass rate.

## Read the scores

| Metric | Meaning |
|---|---|
| Pass/fail | All five positive rubric criteria must be true and `hallucinations` must be false |
| Scenario score | Fraction of the five positive criteria satisfied, minus 0.2 for hallucinations, floored at zero |
| Pass rate | Passed cases divided by graded cases; inspect the graded/attempted counts |
| Rubric success rate | Success frequency for each criterion; hallucination success means **no** hallucination |
| Median/p95 execution time | Completed agent invocation duration, excluding grading |
| Error rates | Invocation failures and tool errors; retained retry attempts remain visible |

A score of **0.8 can still fail** when one positive criterion fails. Inspect the rationale and rubric results before attributing a low pass rate to model quality. The judge sees the final answer plus the first 8,000 characters of the serialized trajectory; full observed traces are saved separately.

## Troubleshooting

| Symptom | Check |
|---|---|
| Claude judge fails | `claude auth status`, access to `claude-fable-5-1`, and CLI support for `--safe-mode` |
| Codex model/flag rejected | Use a compatible authenticated CLI; check the launcher's hardcoded executable path |
| CouchDB connection or authentication fails | Docker initialization logs and `.env` credentials |
| FMSR returns `LLM unavailable` | Its Watsonx configuration; Claude/Codex login does not authenticate that internal tool model |
| Existing output rejected | Use a fresh directory for a new comparison; reuse the single-model command only to resume matching work |
| Snapshot mismatch on repetitions | Restore the baseline's starting database data; do not bypass the hash check |
| No grades yet | Inspect `live-grading.log` and each measurement's `grading.status` |

The [published run report](../benchmarks/runs/2026-09-30-transformer/README.md) records the original environment limitations. [Evaluation reference](evaluation.md) covers schemas and custom scorers; [runner reference](../benchmarks/generated-scenarios.md) covers detailed measurement and database auditing.
