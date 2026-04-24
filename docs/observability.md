# Observability

AssetOpsBench instruments every agent run with OpenTelemetry tracing so each
benchmark invocation produces a durable, standards-based trace record.  The
primary use case is **saving traces as evaluation artifacts**; live UIs like
Jaeger are a secondary nice-to-have.

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
The trajectory attached to `AgentResult` still carries that information
if you need it locally.

## Enabling tracing

OTEL is opt-in.  Install the optional dependency group:

```bash
uv sync --group otel
```

Then point each run at an OTLP endpoint:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  uv run deep-agent --run-id bench-001 --scenario-id 304 \
  "Calculate the bearing characteristic frequencies for a 6205 bearing at 1800 RPM."
```

If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, `init_tracing()` silently falls
back to a no-op tracer — runs work normally with zero overhead.

## Persisting traces to disk (recommended)

Use the Collector config at the repo root.  The `file` exporter writes
OTLP-JSON line-by-line — a vendor-neutral format that any OTLP backend can
ingest later via the `otlpjsonfile` receiver.

```bash
mkdir -p traces
docker run -d --rm --name otel-collector \
  -p 4318:4318 \
  -v "$(pwd)/traces:/traces" \
  -v "$(pwd)/otel-collector.yaml:/etc/otelcol-contrib/config.yaml" \
  otel/opentelemetry-collector-contrib

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  uv run deep-agent --run-id bench-001 "What time is it?"

docker stop otel-collector
```

Each span batch appends one JSON line to `traces/traces.jsonl`.  Query with
`jq`:

```bash
jq -c 'select(.resourceSpans[].scopeSpans[].spans[].attributes[]
              | select(.key == "agent.run_id" and .value.stringValue == "bench-001"))' \
   traces/traces.jsonl
```

The Collector rotates the file (see `rotation` in the config) so long
benchmark runs don't produce one unbounded blob.

## Replaying saved traces into a live backend

Point any OTLP-compatible collector at the on-disk file using the
`otlpjsonfile` receiver, then forward to Jaeger / Tempo / Honeycomb as
normal.  Example Collector config for replay:

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

## Live debugging (optional)

For ad-hoc inspection with Jaeger UI instead of (or in addition to) the
Collector:

```bash
docker run -d --rm --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  uv run deep-agent "$query"

open http://localhost:16686  # macOS
```

Jaeger all-in-one stores traces **in memory**; restart = data gone.  Use the
Collector config above when you care about persistence.

## Troubleshooting

**"OTEL packages not installed; tracing disabled"** — run `uv sync --group otel`.

**No `agent.run_id` on the span** — either `--run-id` wasn't passed and
`set_run_context` was called before a UUID was generated (shouldn't happen
via the CLI; possible if calling runners programmatically), or you're
calling `runner.run()` directly without going through the CLI.  Call
`observability.set_run_context(run_id=...)` yourself before invoking the
runner.

**Exporter silently failing** — set `OTEL_LOG_LEVEL=debug` to see the SDK's
internal logs.
