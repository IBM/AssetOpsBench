# Observability

Each agent run produces two artifacts, joined by ``run_id``:

1. **Trace** — an OpenTelemetry span with *metadata* and *aggregate
   metrics* (runner, model, IDs, latency via span duration, token
   totals, turn and tool-call counts).  Written as canonical OTLP-JSON
   and recognised by every OTEL-aware backend (Jaeger, Tempo, Langfuse,
   Grafana Cloud AI, Honeycomb).
2. **Trajectory** — a per-run JSON file with *per-turn content*: turn
   text, tool call inputs / outputs, and (for SDK runners) per-turn
   token usage.  Written directly by the agent runner alongside the
   trace.

Spans and trajectories complement each other without duplicating
content: the span holds everything an observability UI needs to
summarise or bill a run, the trajectory holds the raw per-turn data
needed for offline evaluation.  Aggregate numbers (totals) live on the
span; per-turn numbers (from which the totals are derived) live on the
trajectory.  Nothing is repeated.

## Root span attributes

Metadata + aggregate metrics — always written when tracing is enabled:

"SDK runners" below means claude-agent, openai-agent, deep-agent (which all
expose turn/tool-call bookkeeping); plan-execute's loop is step-shaped and
surfaces different attributes.

| Attribute                     | Runner coverage   | Notes                                  |
| ----------------------------- | ----------------- | -------------------------------------- |
| `agent.runner`                | all               | `plan-execute` / `claude-agent` / …    |
| `gen_ai.system`               | all               | Provider family (anthropic, openai…)   |
| `gen_ai.request.model`        | all               | Full model ID                          |
| `gen_ai.usage.input_tokens`   | all               | Sum across the run                     |
| `gen_ai.usage.output_tokens`  | all               | Sum across the run                     |
| `agent.question.length`       | all               | Character length of the question       |
| `agent.answer.length`         | all               | Character length of the final answer   |
| `agent.duration_ms`           | all               | Wall-clock of `run()`                  |
| `agent.run_id`                | all               | `--run-id` or auto-generated UUID4     |
| `agent.scenario_id`           | all               | `--scenario-id` (omitted if unset)     |
| `agent.turns`                 | SDK runners       | Number of turns                        |
| `agent.tool_calls`            | SDK runners       | Total tool calls                       |
| `agent.llm_time_ms`           | plan-execute      | Planning + summarisation LLM time      |
| `agent.planning_time_ms`      | plan-execute      | `Planner.generate_plan` wall-clock     |
| `agent.summarization_time_ms` | plan-execute      | Final summarise-LLM wall-clock         |
| `agent.plan.steps`            | plan-execute      | Number of generated plan steps         |

For plan-execute, ``gen_ai.usage.*`` is the run-wide sum across planning,
per-step arg-resolution, and summarisation LLM calls (provided the backend
reports usage — ``LiteLLMBackend`` does; mocks return zero).  Turn and
tool-call counts have no clean mapping to the step-shaped loop and are
not surfaced; per-step wall-clock lives on each ``StepResult.duration_ms``
in the trajectory.

Per-tool timing is not captured for the three SDK runners — the
`PreToolUse` hook that claude-agent needed broke compatibility with
some `@anthropic-ai/claude-code` CLI versions, and the openai / deep
SDKs do not expose clean per-tool callback surfaces either.  Follow-up
when needed.

Plus automatic child spans from the `HTTPXClientInstrumentor` — one per
outbound HTTP request to the LiteLLM proxy (URL, status, latency).  The
root span's own duration = agent wall-clock, so ``agent.duration_ms`` is
redundant for OTEL UIs but convenient for jq on the JSONL file.

## Trajectory file layout

When ``AGENT_TRAJECTORY_DIR`` is set, each runner writes
``{AGENT_TRAJECTORY_DIR}/{run_id}.json``.  The `trajectory` field's shape
depends on the runner.

| Field                         | claude-agent | openai-agent | deep-agent | plan-execute |
| ----------------------------- | ------------ | ------------ | ---------- | ------------ |
| `Trajectory.started_at`       | ✓            | ✓            | ✓          | (n/a)        |
| `TurnRecord.duration_ms`      | ✓            | ✗            | ✗          | (n/a)        |
| `ToolCall.duration_ms`        | ✗            | ✗            | ✗          | (n/a)        |
| `StepResult.duration_ms`      | (n/a)        | (n/a)        | (n/a)      | ✓            |

plan-execute's trajectory is a list of ``StepResult`` records instead
of turns, each carrying its own ``duration_ms`` populated by the executor.

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

For metadata + aggregate metrics (run_id, runner, model, token totals,
latency) read the trace alone — token totals are on the span:

```bash
jq -c '.resourceSpans[].scopeSpans[].spans[]
       | select(.name | startswith("agent.run"))
       | {
           run_id: (.attributes[] | select(.key == "agent.run_id") | .value.stringValue),
           runner: (.attributes[] | select(.key == "agent.runner") | .value.stringValue),
           model: (.attributes[] | select(.key == "gen_ai.request.model") | .value.stringValue),
           input_tokens: (.attributes[] | select(.key == "gen_ai.usage.input_tokens") | .value.intValue),
           output_tokens: (.attributes[] | select(.key == "gen_ai.usage.output_tokens") | .value.intValue),
           turns: (.attributes[] | select(.key == "agent.turns") | .value.intValue),
         }' traces/traces.jsonl
```

For per-turn content (text, tool call inputs/outputs, per-turn tokens)
read the matching trajectory file:

```bash
jq '.trajectory.turns[] | {index, input_tokens, tool_calls: [.tool_calls[].name]}' \
   traces/trajectories/bench-001.json
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
