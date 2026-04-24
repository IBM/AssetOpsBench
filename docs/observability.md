# Observability

AssetOpsBench instruments every agent run with OpenTelemetry tracing so each
benchmark invocation produces a durable, standards-based trace record.  The
primary use case is **saving traces as evaluation artifacts** — no Docker,
no collector, no network dependency.  Live observability (Jaeger / Tempo)
is a secondary mode for teams that want it.

## What gets recorded

One root span per `runner.run(question)` call, tagged with:

| Attribute                     | Source                               |
| ----------------------------- | ------------------------------------ |
| `agent.runner`                | `plan-execute` / `claude-agent` / …  |
| `gen_ai.system`               | Provider family (anthropic, openai…) |
| `gen_ai.request.model`        | Full model ID                        |
| `gen_ai.usage.input_tokens`   | Total across the run                 |
| `gen_ai.usage.output_tokens`  | Total across the run                 |
| `agent.turns`                 | Number of turns                      |
| `agent.tool_calls`            | Number of tool calls                 |
| `agent.question.length`       | Character length of the question     |
| `agent.answer.length`         | Character length of the final answer |
| `agent.run_id`                | `--run-id` or auto-generated UUID4   |
| `agent.scenario_id`           | `--scenario-id` (omitted if unset)   |

Plus automatic child spans from the `HTTPXClientInstrumentor` — one per
outbound HTTP request to the LiteLLM proxy (URL, status, latency).

**Not recorded**: raw prompt / response text, per-turn tool inputs / outputs.
The trajectory on `AgentResult` still carries that information in-process.

## Enabling tracing

Install the optional dependency group:

```bash
uv sync --group otel
```

Tracing activates when either of these env vars is set:

| Env var                           | Effect                                              |
| --------------------------------- | --------------------------------------------------- |
| `OTEL_TRACES_FILE`                | Append OTLP-JSON lines to this path (in-process).   |
| `OTEL_EXPORTER_OTLP_ENDPOINT`     | Ship spans over HTTP to a live collector endpoint.  |

Both can be set simultaneously.  If neither is set, `init_tracing()` is a
no-op and runs work normally with zero overhead.

## Saving traces to disk (recommended)

```bash
OTEL_TRACES_FILE=./traces/traces.jsonl \
  uv run deep-agent --run-id bench-001 --scenario-id 304 \
  "Calculate bearing characteristic frequencies for a 6205 bearing at 1800 RPM."
```

Each span batch appends one JSON line to `./traces/traces.jsonl` in
canonical OTLP-JSON format — the same format the OpenTelemetry Collector's
`file` exporter produces, and ingestible by the Collector's
`otlpjsonfile` receiver later if you want to replay into a live backend.

### Query with `jq`

```bash
# All spans for a particular run
jq -c '.resourceSpans[].scopeSpans[].spans[]
       | select(.attributes[]
                | select(.key == "agent.run_id" and .value.stringValue == "bench-001"))' \
   traces/traces.jsonl

# Token totals per run
jq -c '.resourceSpans[].scopeSpans[].spans[]
       | select(.name | startswith("agent.run"))
       | {
           run: (.attributes[] | select(.key == "agent.run_id") | .value.stringValue),
           model: (.attributes[] | select(.key == "gen_ai.request.model") | .value.stringValue),
           input: (.attributes[] | select(.key == "gen_ai.usage.input_tokens") | .value.intValue),
           output: (.attributes[] | select(.key == "gen_ai.usage.output_tokens") | .value.intValue),
         }' \
   traces/traces.jsonl
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

**Exporter silently failing** — set `OTEL_LOG_LEVEL=debug` for the SDK's
internal logs.
