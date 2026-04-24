# Observability

Each agent run produces two artifacts, joined by ``run_id``:

1. **Trace** — an OpenTelemetry span with *metadata* (runner, model, IDs,
   timing, outbound HTTP calls to the LiteLLM proxy).  Written as
   canonical OTLP-JSON, replayable into any OTLP backend.
2. **Trajectory** — a per-run JSON file with *content* (per-turn text,
   tool call inputs / outputs, per-turn token usage).  Written
   directly by the agent runner.

Spans and trajectories intentionally **do not overlap**: spans don't
carry token totals or turn counts (which would duplicate what's in the
trajectory), and trajectories don't carry runner/model metadata
(which is the span's job).  To reconstruct a full picture of a run,
join by ``run_id``.

## Root span attributes

| Attribute                   | Notes                                 |
| --------------------------- | ------------------------------------- |
| `agent.runner`              | `plan-execute` / `claude-agent` / …   |
| `gen_ai.system`             | Provider family (anthropic, openai…)  |
| `gen_ai.request.model`      | Full model ID                         |
| `agent.question.length`     | Character length of the question      |
| `agent.answer.length`       | Character length of the final answer  |
| `agent.run_id`              | `--run-id` or auto-generated UUID4    |
| `agent.scenario_id`         | `--scenario-id` (omitted if unset)    |
| `agent.plan.steps`          | *plan-execute only*                   |

Plus automatic child spans from the `HTTPXClientInstrumentor` — one per
outbound HTTP request to the LiteLLM proxy (URL, status, latency).

## Trajectory file layout

When ``AGENT_TRAJECTORY_DIR`` is set, each runner writes
``{AGENT_TRAJECTORY_DIR}/{run_id}.json``:

```json
{
  "run_id": "bench-001",
  "scenario_id": "304",
  "runner": "deep-agent",
  "model": "litellm_proxy/aws/claude-opus-4-6",
  "question": "...",
  "answer": "...",
  "trajectory": {
    "turns": [
      {
        "index": 0,
        "text": "",
        "tool_calls": [{"name": "sensors", "input": {...}, "output": {...}}],
        "input_tokens": 14248,
        "output_tokens": 41
      },
      ...
    ]
  }
}
```

plan-execute's trajectory is a list of ``StepResult`` records instead
of turns; the structure is otherwise analogous.

## Enabling persistence

Install the optional tracing deps (trajectories need no extra deps):

```bash
uv sync --group otel
```

Each artifact has its own env var; set either, both, or neither:

| Env var                           | Effect                                              |
| --------------------------------- | --------------------------------------------------- |
| `AGENT_TRAJECTORY_DIR`            | Directory for ``{run_id}.json`` trajectory records. |
| `OTEL_TRACES_FILE`                | Append OTLP-JSON lines to this path (in-process).   |
| `OTEL_EXPORTER_OTLP_ENDPOINT`     | Ship spans over HTTP to a live collector endpoint.  |

When none are set, runs work normally with zero persistence overhead.

## Recommended: save both traces and trajectories

```bash
AGENT_TRAJECTORY_DIR=./traces/trajectories \
OTEL_TRACES_FILE=./traces/traces.jsonl \
  uv run deep-agent --run-id bench-001 --scenario-id 304 \
  "Calculate bearing characteristic frequencies for a 6205 bearing at 1800 RPM."
```

Each span batch appends one JSON line to `./traces/traces.jsonl` in
canonical OTLP-JSON format — the same format the OpenTelemetry Collector's
`file` exporter produces, and ingestible by the Collector's
`otlpjsonfile` receiver later if you want to replay into a live backend.

### Query with `jq`

Use the trace for metadata queries (which model, which runner, how long);
use the trajectory for content queries (token totals, per-turn detail,
tool call arguments):

```bash
# List run metadata from traces
jq -c '.resourceSpans[].scopeSpans[].spans[]
       | select(.name | startswith("agent.run"))
       | {
           run: (.attributes[] | select(.key == "agent.run_id") | .value.stringValue),
           runner: (.attributes[] | select(.key == "agent.runner") | .value.stringValue),
           model: (.attributes[] | select(.key == "gen_ai.request.model") | .value.stringValue),
         }' traces/traces.jsonl

# Token totals across trajectories (sums per-turn usage)
for f in traces/trajectories/*.json; do
  jq -c '{
    run_id,
    input: ([.trajectory.turns[].input_tokens] | add),
    output: ([.trajectory.turns[].output_tokens] | add),
  }' "$f"
done
```

### Rotation

The built-in file exporter appends indefinitely — one line per span batch
is small, but long-running benchmarks can grow.  For rotation, pipe the
path through `logrotate`, or split runs across dated files:

```bash
OTEL_TRACES_FILE="./traces/$(date +%F).jsonl" uv run deep-agent "..."
```

## Replaying saved traces into a live backend (optional)

If you later want to visualize persisted traces, point any
OpenTelemetry Collector at the file with its `otlpjsonfile` receiver:

```yaml
receivers:
  otlpjsonfile:
    include: ["traces/traces.jsonl"]
exporters:
  otlp:
    endpoint: jaeger:4317
    tls: {insecure: true}
service:
  pipelines:
    traces:
      receivers: [otlpjsonfile]
      exporters: [otlp]
```

## Live debugging with Jaeger (optional, Docker)

When network access to Docker Hub is available, Jaeger all-in-one is the
quickest way to inspect traces in a UI:

```bash
docker run -d --rm --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_TRACES_FILE=./traces/traces.jsonl \
  uv run deep-agent --run-id demo "$query"

open http://localhost:16686   # macOS
```

With both env vars set, spans go to disk *and* to Jaeger simultaneously.
Jaeger all-in-one is in-memory only; the file stays on disk when the
container exits.

## Troubleshooting

**"OTEL SDK not installed; tracing disabled"** — run `uv sync --group otel`.

**No output file on disk** — tracing is lazy; at least one runner has to
complete a `run()` call before the `BatchSpanProcessor` flushes.  For small
smoke tests, make sure the CLI exits cleanly (the `atexit` hook flushes
any buffered spans).

**Spans exist but `agent.run_id` is missing** — you called `runner.run()`
programmatically without going through a CLI.  Seed it yourself:

```python
from observability import init_tracing, set_run_context
init_tracing("my-harness")
set_run_context(run_id="...", scenario_id="...")
await runner.run(question)
```

**No trajectory file in `AGENT_TRAJECTORY_DIR`** — the runner skips
persistence when no `run_id` is set.  Use the CLI (which seeds a UUID4
automatically), or call `set_run_context(run_id=...)` before invoking
the runner programmatically.

**Exporter silently failing** — set `OTEL_LOG_LEVEL=debug` for the SDK's
internal logs.
