# Stirrup Agent

The `stirrup-agent` runner drives [Artificial Analysis' Stirrup](https://github.com/ArtificialAnalysis/Stirrup) framework against the AssetOpsBench MCP servers. It is a peer to `plan-execute`, `claude-agent`, `openai-agent`, and `deep-agent`: same CLI contract, same persisted `Trajectory`, scored by the same `uv run evaluate`.

Unlike the other runners, Stirrup is a **code-capable** agent — it can write and execute Python to solve a task, in addition to calling the domain MCP tools. That introduces a **code track** distinct from the tools-only runners.

## Contents

- [Why Stirrup](#why-stirrup)
- [Install](#install)
- [Quick start (tools-only)](#quick-start-tools-only)
- [Model routing](#model-routing)
- [Code execution tracks](#code-execution-tracks)
- [Docker backend](#docker-backend)
- [Reading tool-produced files](#reading-tool-produced-files)
- [CLI flags](#cli-flags)
- [Domain-vs-code routing metric](#domain-vs-code-routing-metric)
- [Validation runs](#validation-runs)
- [Troubleshooting](#troubleshooting)
- [What was added](#what-was-added)

---

## Why Stirrup

- **In-process Python library**, so it integrates like `deep-agent` (no subprocess, no session-file parsing). The runner maps Stirrup's returned message history straight into the shared `Trajectory`.
- **Native LiteLLM client**, so `watsonx/...` and other `<provider>/<model>` strings work directly; the `litellm_proxy/` prefix is also supported.
- **Code execution** through local or Docker backends.
- **MCP client support**, so it connects to the same six AssetOpsBench servers as every other runner.

---

## Install

The dependency and entry point are declared in `pyproject.toml`
(`stirrup[mcp,litellm,docker]` and the `stirrup-agent` script). From the repo root:

```bash
uv sync
```

Confirm the entry point resolved (no model call):

```bash
uv run stirrup-agent --help
```

Run the unit tests for the trajectory mapping:

```bash
uv run pytest src/agent/stirrup_agent/tests/ -q
```

---

## Quick start (tools-only)

Start with `--no-code` — it needs no Docker, so it isolates "does the MCP wiring
work" from "does code execution work." Set your model credentials, then:

```bash
export WATSONX_APIKEY=...  WATSONX_PROJECT_ID=...  WATSONX_URL=https://us-south.ml.cloud.ibm.com

uv run stirrup-agent --no-code --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "What sites are available?"
```

In the `--show-trajectory` output you should see a real domain tool call
(e.g. `iot__sites`) and a sensible answer. That confirms the `{server}__{tool}`
wiring and the trajectory parse.

> **Startup warnings.** Stirrup prints three: local filesystem access (only on
> `--code-backend local`), "Missing default tool: WebToolProvider" (intentional —
> web tools are deliberately not attached so they cannot contaminate the benchmark),
> and "no output_dir set" (files the agent *creates* are discarded; harmless for
> computational answers). None is an error.

### Verify all six MCP servers registered

Tool registration is independent of whether the model used a tool. To confirm
every server connected, list the provider's tools directly:

```bash
uv run python - <<'PY'
import asyncio
from agent.stirrup_agent.runner import StirrupAgentRunner

async def main():
    provider = StirrupAgentRunner(code_enabled=False)._build_mcp_provider()
    async with provider.connect() as p:
        names = sorted(t.name for t in p.get_all_tools())
        by = {}
        for n in names:
            by.setdefault(n.split("__", 1)[0], []).append(n)
        for srv, ts in by.items():
            print(f"{srv}: {len(ts)}")
asyncio.run(main())
PY
```

Expected default counts: `iot` 12, `utilities` 6, `fmsr` 3, `tsfm` 41, `wo` 15, `vibration` 8
(85 MCP tools total; the agent additionally has Stirrup's own `finish` tool). If `AOB_READONLY=1`
is set, `wo` exposes 9 read tools instead of 15.

---

## Model routing

The runner's default model is
`watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8`.

| `--model-id` prefix     | Client                          | Notes                                                       |
| ----------------------- | ------------------------------- | ----------------------------------------------------------- |
| `<provider>/<model>`    | Stirrup `LiteLLMClient`         | Native LiteLLM. `watsonx/...`, `anthropic/...`, etc. work directly. |
| `litellm_proxy/` or `tokenrouter/` | `ChatCompletionsClient` | Uses only the OpenAI-compatible Chat Completions API. |

Required env vars match the rest of the repo: the standard watsonx vars for the
native route, `LITELLM_BASE_URL` / `LITELLM_API_KEY` for the LiteLLM proxy, or
`TOKENROUTER_BASE_URL` / `TOKENROUTER_API_KEY` for TokenRouter.

For context management, the runner gives Stirrup a 100,000-token working-context
budget while leaving each underlying client's 64,000 maximum-output-token default
unchanged. The runner requests summarization at 75% of that budget, or
approximately 75,000 tokens. This budget controls context compaction and is not
model metadata supplied by TokenRouter. When summarization occurs, the complete
generated summary is printed during the run rather than the truncated preview
used by Stirrup's default logger.

---

## Code execution tracks

| Flag                      | Behaviour                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| `--code-enabled` (default)| Adds a code-execution tool. The agent may solve a scenario by writing code. **Code track.** |
| `--no-code`               | Tools only. Directly comparable to `claude-agent` / `openai-agent` / `deep-agent`.          |

A code-enabled run and a tools-only run are **not** comparable 1:1 — report them
on separate leaderboard tracks.

Backends (`--code-backend`):

| Backend  | Isolation | Host file access                | When to use                                       |
| -------- | --------- | ------------------------------- | ------------------------------------------------- |
| `docker` | full      | none (container filesystem)     | unattended runs; the default                      |
| `local`  | none      | reads host paths directly       | development / trusted inputs; fastest, no Docker  |

> `local` runs model-authored code on your host with your permissions. Use it for
> inputs you control; prefer `docker` for unattended or untrusted runs.

When code execution is enabled, the runner appends backend-specific guidance to
the shared agent prompt. It directs the model to answer directly when domain
knowledge, MCP results, or basic reasoning are sufficient; reserve `code_exec`
for necessary computation, data processing, workspace inspection, or validation;
and never use it for planning, comments, placeholders, or empty scripts. When
code is needed, the prompt prefers one bounded inspect/analyze/verify script and
small outputs. Docker runs identify `/workspace` and the installed NumPy, pandas,
and SciPy packages; local runs warn that commands execute with the current user's
host permissions. `--no-code` keeps the shared prompt unchanged.

---

## Docker backend

### 1. Daemon and socket

The Docker Python SDK must reach the daemon. On macOS with Rancher Desktop the
socket is not at the default path, so point the SDK at it (find it with
`docker context inspect | grep Host`):

```bash
export DOCKER_HOST=unix:///Users/<you>/.rd/docker.sock   # Rancher Desktop
docker info                                              # confirm daemon is up
uv run python -c "import docker; print(docker.from_env().version()['Version'])"
```

Rancher Desktop must use the **dockerd (moby)** engine, not containerd.

### 2. Sandbox image with the scientific stack

The default image name is `assetops-code`. Build it once from the bundled
Dockerfile, which adds NumPy, pandas, and SciPy to `python:3.12-slim`:

```bash
docker build -f src/agent/stirrup_agent/Dockerfile.code -t assetops-code .
```

Set `STIRRUP_CODE_IMAGE` only when using a different tag.

`Dockerfile.code`:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    "numpy>=1.24" "pandas>=2.0" "scipy>=1.10"
```

### 3. Run

```bash
export DOCKER_HOST=unix:///Users/<you>/.rd/docker.sock
export STIRRUP_CODE_IMAGE=assetops-code

uv run stirrup-agent --code-backend docker --show-trajectory \
  --workspace-dir /tmp/assetopsbench-stirrup/smoke \
  --preserve-workspace \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "Run python to compute the factorial of 12 and tell me the result"
```

The first run pulls/builds the image, so expect a delay. Look for a `code_exec`
call and the right answer (479001600).

---

## Reading tool-produced files

On the code track, MCP text results larger than 100 KiB (102,400 UTF-8 bytes) are
automatically written under `mcp_results/` in the active code workspace. The
tool response returned to the model is a compact JSON handle containing the
relative path, tool arguments, byte count, and SHA-256 digest. `code_exec` can
read that path directly without copying the original response through another
model turn.
The file contains the complete, unmodified MCP response—including null fields—
rather than a projected subset. Code should inspect only the schema, counts, a
small sample, or the specific rows or fields needed, then process the artifact
in place. For artifacts larger than 200 KiB, extract and process the relevant
subset in bounded batches instead of printing the full payload.

Within one agent run, identical read calls reuse an intact existing artifact.
Successful work-order mutations, catalog mutations, and TSFM run-producing tools
clear the read cache so later reads observe their changes. This is a run-local
snapshot cache; it does not detect updates made externally during the run.
Artifacts are content-addressed, so a refreshed response does not overwrite an
earlier snapshot. Smaller responses remain inline, and persistence failures
safely fall back to the original inline MCP result.

For example:

```bash
python3 - <<'PY'
import json

with open("mcp_results/wo__list_workorders_<query-id>_<content-id>.json") as f:
    result = json.load(f)

print(len(result["work_orders"]))
PY
```

The workspace is temporary unless `--preserve-workspace` is enabled.

---

## Preserving code workspaces

By default, Stirrup creates a temporary execution directory for `code_exec` and
removes it when the agent session exits. Pass `--workspace-dir` to choose the host
base directory for that sandbox, and pass `--preserve-workspace` to copy the final
code-execution files back into that directory before cleanup.

For scenario-suite runs, use a root outside the repo:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-root /path/to/scenarios_data \
  --agent_name stirrup_agent \
  --stirrup-workspace-root /tmp/assetopsbench-stirrup-workspaces \
  --preserve-workspaces
```

For scenario `401`, the preserved files are available under a nested path such as:

```text
/tmp/assetopsbench-stirrup-workspaces/stirrup_agent/tokenrouter-MiniMax-M3/stirrup_agent_401/
```

---

## CLI flags

In addition to the [common flags](../INSTRUCTIONS.md#common-flags) (`--model-id`,
`--show-trajectory`, `--json`, `--run-id`, `--scenario-id`, ...):

| Flag                  | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| `--code-enabled`      | Enable code execution (default). The code track.                                     |
| `--no-code`           | Tools-only; comparable to the other runners.                                         |
| `--code-backend`      | `docker` (default) or `local`.                                                        |
| `--topology`          | `flat` (default) or `gateway`. See [Tool-surface topology](#tool-surface-topology).   |
| `--gateway-mode`      | `index` (default) or `search`, under `--topology gateway`.                           |
| `--gateway-top-k K`   | Candidates returned by the gateway's `search_tools` (default: 3).                     |
| `--max-turns N`       | Max agent turns (default: 30).                                                       |
| `--reasoning-effort LEVEL` | Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or `default`); provider default when omitted. |
| `--workspace-dir PATH` | Host base directory for Docker/local code-execution workspaces.                     |
| `--preserve-workspace` | Copy final code-execution files into `--workspace-dir` before cleanup.              |

Environment variable: `STIRRUP_CODE_IMAGE` (Docker image; default `assetops-code`).

---

## Domain-vs-code routing metric

Because a code-capable agent can bypass the domain tools (solve by writing code
instead of calling `wo__*` / `tsfm__*` / ...), every run records how its tool calls
were routed, as span attributes on the run:

| Attribute                  | Meaning                                                  |
| -------------------------- | -------------------------------------------------------- |
| `agent.code_track`         | Whether code execution was enabled.                      |
| `agent.domain_tool_calls`  | Count of MCP (`{server}__{tool}`) calls.                 |
| `agent.code_tool_calls`    | Count of `code_exec` calls.                              |
| `agent.tool_bypass`        | `true` if it used code but **no** domain tools.          |

This quantifies how often the agent abandoned the grounded tools — a reportable
measurement, not a bug. See [docs/observability.md](observability.md) for reading spans.

---

## Tool-surface topology

Attaching every MCP server to one agent pins the whole tool manifest into the
context on every turn, whether or not a scenario touches those domains.
`--topology` controls that, orthogonally to the code track, and is recorded on
every run as `agent.topology`.

**`flat` (default).** Every registered server is attached directly. This is the
shape `claude-agent`, `openai-agent` and `deep-agent` use, so it is the
configuration to report against them.

**`gateway`.** Every server sits behind three routing tools, so the agent sees a
handful of entries instead of the full catalogue while keeping one context and
one trajectory:

- `search_tools(query, k)` ranks the catalogue by BM25 and returns the top `k`
  as `{name, server, summary}`.
- `describe_tools(names)` returns full parameter schemas on demand.
- `call_tool(name, arguments)` validates against the real tool's model and runs it.

Two modes separate two different savings. `--gateway-mode index` pins a compact
one-line-per-tool catalogue into `describe_tools`' description, so it is re-sent
every turn: the agent always knows what exists and pays only for schema
deferral. `--gateway-mode search` withholds the catalogue too, so discovery
becomes a hard dependency on retrieval quality. Reporting both isolates how much
of any saving comes from deferring schemas versus from withholding the
catalogue, which one mode alone cannot tell you.

Retrieval is lexical BM25, implemented in-repo with no model and no network.
That is a reproducibility decision: a benchmark arm whose retrieval quality is
an unpinned variable cannot be replayed. Ranking ties break on tool name, so a
run is byte-reproducible. The BM25 constants are fixed rather than tuned, since
tuning them against the scenarios would leak the evaluation set into the routing
layer.

Three implementation notes worth knowing:

1. **The gateway wraps whichever provider the track built.** On the code track
   that is `WorkspaceBridgedMCPToolProvider`, so oversized results keep spilling
   into `mcp_results/` exactly as under `flat`. It works on the no-code track
   too, so the gateway can be compared against the flat baseline on the same
   track as the other runners.
2. **`MCPGatewayToolProvider` must subclass Stirrup's `ToolProvider`.**
   `Agent.__init__` decides what to connect with `isinstance(t, ToolProvider)`,
   so a gateway that did not subclass it would silently never connect and the
   agent would start with no domain tools. `test_gateway.py` asserts this.
3. **Argument errors are recoverable here.** The gateway's own parameters always
   validate, so a bad inner argument reaches our executor rather than being
   replaced by Stirrup's fixed `"Tool arguments are not valid"`, whose real
   reason goes only to a debug log. `call_tool` returns the actual validation
   error plus the schema, so the agent can correct itself in one turn instead of
   reissuing the same call until `max_turns`.

### Running it from the benchmark

`benchmarks/run_tiny.sh` selects the topology by method name:

```bash
AGENTS="stirrup_agent stirrup_agent_gateway stirrup_agent_gateway_search" \
  ./benchmarks/run_tiny.sh "$SCENARIO_DIR" "$LEADERBOARD_DIR"
```

The three are the same runner, model and scenarios; only the tool surface
differs. They are separate method names deliberately: `scenario_suite_runner`
nests outputs as `<root>/<agent_name>/<model>/` and names runs
`{agent_name}_{scenario_id}`, so a shared name would have each run overwrite the
last. `--agent_name all` now includes both gateway variants.

### What to measure

Gateway runs record `agent.gateway_mode`, `agent.gateway_discovery_calls`,
`agent.gateway_tool_count` and `agent.gateway_schemas_disclosed`. Domain calls
are still credited to the underlying server, because `classify_tool` reads the
`name` argument of a `call_tool` invocation, so `agent.domain_tool_calls` and
`agent.tool_bypass` stay comparable with a flat run.

The topology trades total tokens for context headroom, and those move in
opposite directions. Report both:

- `agent.peak_context_tokens` should fall sharply, since the agent carries three
  routing tools instead of every schema.
- `gen_ai.usage.input_tokens` may rise, because discovery costs turns and
  described schemas re-appear in history.
- `agent.gateway_discovery_calls` against `agent.domain_tool_calls` is the
  overhead ratio, and the number that decides whether the topology pays for
  itself.

`scripts/measure_topology_context.py` reports the per-turn static cost of every
topology against the real servers without calling a model.

---

## Validation runs

A reproducible checklist a teammate can follow to confirm a working setup:

```bash
# 0. install + entry point
uv sync
uv run stirrup-agent --help

# 1. unit tests (no services)
uv run pytest src/agent/stirrup_agent/tests/ -q

# 2. all six MCP servers register  (see "Verify all six MCP servers" above)

# 3. tools-only smoke (no Docker)
uv run stirrup-agent --no-code --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "What sites are available?"

# 4. local code track (no Docker)
uv run stirrup-agent --code-backend local --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "Compute the factorial of 12 using python"

# 5. docker code track  (after building assetops-code + setting DOCKER_HOST)
uv run stirrup-agent --code-backend docker --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "Run python to compute the factorial of 12 and tell me the result"

# 6. persist
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories
uv run stirrup-agent --no-code --run-id stirrup-smoke --scenario-id 101 \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "List all failure modes of asset Chiller."
```

---

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `docker.errors.DockerException: Error while fetching server API version ... FileNotFoundError` | SDK can't find the daemon socket. `export DOCKER_HOST=unix://<path from 'docker context inspect | grep Host'>`. |
| Docker connects but `code_exec` hits `ModuleNotFoundError` | Sandbox image lacks the library; build/point `STIRRUP_CODE_IMAGE` at `assetops-code` (or an image with the stack). |
| Agent code cannot open a server-returned dataset or result file pointer in Docker | Those host-side pointers are separate from automatically bridged `mcp_results/` snapshots. Use an MCP tool that consumes the pointer, or use `--code-backend local` when direct host-file access is required. |
| `uv sync` fails to resolve | A pin (e.g. `litellm==...`) clashing with Stirrup's range; relax to a compatible range. |
| `stirrup` import errors | `stirrup[mcp,litellm,docker]` not installed; re-run `uv sync`. |
| A server missing from the tool list | Its subprocess failed to start (missing CouchDB creds for `iot`/`wo`/`vibration`, or model load for `tsfm`). |

---

## What was added

- `src/agent/stirrup_agent/` — runner, trajectory mapping, workspace bridging,
  CLI, Docker image, and focused tests.
- `src/agent/_prompts.py` — concise enforcement of user-requested response formats.
- `pyproject.toml` — `stirrup[mcp,litellm,docker]` dependency and the `stirrup-agent`
  entry point.

The MCP servers themselves are unchanged.
