# Comparing agents on generated scenarios

Use `python -m benchmark.generated_suite_runner` from the repository with
`PYTHONPATH=src`. The runner requires a completed generation manifest and the
exact requested positive/negative counts. It reads both scenario JSON files,
runs tool-enabled agents, and saves separate trajectories and logs per target.
It never resets or reloads the live databases.

Start with `--dry-run`, then `--limit 1` to verify model access and tool calls.
Remove those flags for the full suite. Repeating a command resumes matching
saved scenarios. Changing the model, harness or generation run under an existing
target name is rejected. Failed calls stop the target and retain its logs.

```bash
# Replace RUN_DIR with the completed generated run directory.
PYTHONPATH=src .venv/bin/python -m benchmark.generated_suite_runner RUN_DIR \
  --output-dir generated/comparisons/transformer --name opus-5-5 \
  --agent claude --model-id claude-opus-5-5 --limit 1

# Uses the locally authenticated Codex subscription.
PYTHONPATH=src .venv/bin/python -m benchmark.generated_suite_runner RUN_DIR \
  --output-dir generated/comparisons/transformer --name gpt-6-astra \
  --agent codex --model-id gpt-6-astra --limit 1

PYTHONPATH=src .venv/bin/python -m benchmark.generated_suite_runner RUN_DIR \
  --output-dir generated/comparisons/transformer --name glm-5-3 \
  --agent openai --model-id zai/glm-5.3 --limit 1
```

Opus uses Claude Agent SDK with Claude Code authentication. Bare OpenAI model
IDs with `--agent openai` use `OPENAI_API_KEY`. The `--agent codex` runner uses
your existing `codex login` subscription, exposes the benchmark MCP servers,
and saves standard trajectories for the existing evaluator.
Proxy model IDs may instead use `litellm_proxy/` or `tokenrouter/` with the
corresponding credentials. GLM uses `ZAI_API_KEY` and optional `ZAI_BASE_URL`.
Set `GLM_REASONING_EFFORT=low` (or `high` / `max`) for explicit GLM reasoning.
Record this setting separately for each comparison target.
The CLI loads local `.env`; credentials must never go in this document.

The examples compare different SDK harnesses. For a model comparison under a
single harness, use `--agent openai` for every target and an OpenAI-compatible
proxy route for Opus. Verify each exact model route with the one-scenario smoke
run before launching the full comparison.

To score, use the same judge across all targets. Add
`--judge-model MODEL_ID` to each full-run command, using an independent judge session. Same-model judging is rejected by default;
`--allow-self-judge` explicitly permits it with a separate session. Alternatively invoke
`python -m evaluation.cli` on its trajectory directory with both scenario files.
Native Claude Code and `zai/` model IDs are supported as judges. A judge score
uses the generated characteristic behavior as its rubric; it is not independently
verified ground truth.

Open-form work-order tools can modify database state. Compare against equivalent
starting data for every target. This runner does not restore snapshots; establish
the benchmark database reset/snapshot policy before a full comparison. Review
logs for tool failures, not only final scores.

## Fresh-run measurements

All targets now measure the entire child-agent invocation, including CLI/SDK
startup, MCP connection, model and tool work, persistence, cleanup and exit.
The record is written before launch and finalized for completed, failed,
timed-out or cancelled invocations. Retries retain separate attempt files.
No model runs are started by reporting or configuration commands.

Each target writes:

- `settings.json`: exact model/provider/harness, installed CLI/SDK versions,
  suite and rubric hashes, reasoning setting, limits, timeout/retry policy,
  database policy, concurrency and order. Unknown provider defaults are null.
- `measurements/<run-id>.json`: start/end/duration, run status/error,
  observed tokens and tool statistics, and separately timed grading outcomes.
- `traces/<run-id>.jsonl`: incrementally saved full observed messages,
  request inputs/responses, tool arguments/results/errors and UTC timestamps.
  Traces survive interrupted invocations. Judge traces have a `.judge` suffix.
- `trajectories/<run-id>.json`: the established evaluator input format.

The five target definitions are in `benchmarks/generated-comparison.json`.
Fable 5.1 judges each model in a fresh, tool-free Claude Code session. Pass
`--allow-self-judge` for the Fable execution target, as explicitly requested;
this permits a separate judge session using the same model. `--judge-model`
uses measured per-scenario grading rather than a second timing aggregate.
Set GLM explicitly with `--reasoning-effort low`. Record total simultaneous
execution targets using `--concurrency-level N`; each target remains serial.

First-response timing means the first observed assistant/protocol response,
not first-token latency. OpenAI-compatible SDK request timing includes its
internal retry handling. Claude/Codex do not expose all underlying request
boundaries, retry counts, reasoning-token subdivisions or compactions; those
fields remain null when unavailable. CLI token costs are provider-reported
estimates, not actual subscription charges. Actual billed API cost is null
unless a billing source supplies it; no guessed prices are used in this report.

For authoritative database-write auditing, use an **existing isolated**
`eval_...` CouchDB namespace with `python -m benchmark.database_audit
--prefix PREFIX --port PORT --audit-dir DIR`. It reads upstream connection
credentials from `.env`. Set `BENCHMARK_DB_PROXY_URL` to its localhost URL and
`BENCHMARK_DB_AUDIT_DIR` to the same directory before executing a target. The
standalone path mode adds the run ID to the proxy path; writes and record changes
are joined by that ID. For IoT, use the root-URL mode described below. Without auditing, database action metrics remain null. Store the
snapshot hash, source, reset policy and proxy namespace in the target's
`environment.json` before launch. This helper does not reset the live database.

Generate comparisons from measured runs only:

```bash
PYTHONPATH=src .venv/bin/python -m benchmark.comparison_report \
  generated/comparisons/transformer --output generated/comparisons/transformer/comparison.html
```

Pass rates and rubric success rates use graded cases, with observed denominators
shown. Median/p95 execution time uses completed invocations with measured time;
failed and cancelled attempts remain in reliability counts. Missing token and
tool values are excluded from means and totals, and availability counts are
included. Historical timing is never reconstructed.

The configured five-model comparison can be launched with:

```bash
PYTHONPATH=src .venv/bin/python tools/run_generated_comparison.py
```

This captures one common source snapshot, clones a namespace per target, checks
connectivity with the actual IoT CouchDB client, runs the targets concurrently,
and grades each target with independent Fable sessions. For the local Codex
subscription, it uses the app's bundled CLI; the older CLI on PATH rejects
GPT-6.1 Sol. The measured records preserve the exact CLI version used.

The comparison launcher uses one root-URL proxy per target and an active-run
file to attribute writes, because the IoT `couchdb3` client discards URL paths.
`BENCHMARK_DB_RUN_ID_FILE` tells the suite runner to update that attribution
before each invocation. Authentication requests do not count as database writes.
The standalone path-based proxy mode is intended for clients that retain paths.

Live view:

```bash
PYTHONPATH=src .venv/bin/python tools/live_evaluation/server.py
```

Open `http://127.0.0.1:8765`. The final offline HTML shares the live view's layout.

## Independent repetitions

Keep the published transformer comparison as repetition 1 and execute two more:

```bash
PYTHONPATH=src .venv/bin/python tools/run_repeated_comparison.py \
  --output-dir generated/comparisons/transformer-k3

PYTHONPATH=src .venv/bin/python tools/live_evaluation/server.py --port 8766 \
  --experiment generated/comparisons/transformer-k3/experiment.json
```

Repetitions run serially; the five model targets and independent grading workers
run concurrently within each repetition. The launcher verifies the reference
suite and initial database hash, preserves one snapshot, and clones fresh isolated
namespaces for each repetition/model. Scenario order, harnesses, reasoning,
timeouts and judge remain the same. Each measurement records its repetition index.

When all three repetitions have finished:

```bash
uv run tools/publish_repeated_comparison.py \
  --experiment generated/comparisons/transformer-k3/experiment.json
```

The report shows equal-weight means and sample standard deviations across full
repetitions, individual repetition results, pooled metrics and per-scenario pass
frequencies. Failed/retried invocation attempts remain in reliability metrics;
they are not additional repetitions. Missing metrics remain missing. The live
view uses only fully completed repetitions for its averages. The publisher checks
suite/settings consistency and distinct judge sessions before emitting a final
comparison, graphs, CSV/JSON data and portable compressed traces.
