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
- **Sandboxed code execution** out of the box (local / Docker / E2B).
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

Expected counts: `iot` 4, `utilities` 3, `fmsr` 2, `tsfm` 6, `wo` 8, `vibration` 8
(31 MCP tools total; the agent additionally has Stirrup's own `finish` tool).

---

## Model routing

The runner's default model is
`watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8`.

| `--model-id` prefix     | Client                          | Notes                                                       |
| ----------------------- | ------------------------------- | ----------------------------------------------------------- |
| `<provider>/<model>`    | Stirrup `LiteLLMClient`         | Native LiteLLM. `watsonx/...`, `anthropic/...`, etc. work directly. |
| `litellm_proxy/<model>` | Stirrup `ChatCompletionsClient` | Points at the LiteLLM proxy (OpenAI-compatible).            |

Required env vars match the rest of the repo: the standard watsonx vars for the
native route, or `LITELLM_BASE_URL` / `LITELLM_API_KEY` for the proxy route.

> **Output-token cap.** Stirrup forwards `--max-tokens` as the completion's
> *max output tokens*. watsonx caps new tokens at 100k, so keep `--max-tokens`
> below that. Default is `16384`.

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
| `e2b`    | full      | remote sandbox                  | hosted execution                                  |

> `local` runs model-authored code on your host with your permissions. Use it for
> inputs you control; prefer `docker` for unattended or untrusted runs.

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

`python:3.12-slim` has no numpy/pandas/scipy. Build the bundled image, which adds them:

```bash
docker build -f src/agent/stirrup_agent/Dockerfile.code -t assetops-code .
export STIRRUP_CODE_IMAGE=assetops-code
```

`Dockerfile.code`:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    "numpy>=1.24" "pandas>=2.0" "scipy>=1.10" "matplotlib>=3.7"
```

### 3. Run

```bash
export DOCKER_HOST=unix:///Users/<you>/.rd/docker.sock
export STIRRUP_CODE_IMAGE=assetops-code

uv run stirrup-agent --code-backend docker --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  "Run python to compute the factorial of 12 and tell me the result"
```

The first run pulls/builds the image, so expect a delay. Look for a `code_exec`
call and the right answer (479001600).

---

## Reading tool-produced files

The Docker sandbox has its own filesystem: it can only see what is mounted, which
is a private working directory — **not** wherever a tool downloaded a file on the
host. So if a tool (e.g. `iot__history`) writes a file on the host and returns its
path, code running in the Docker backend cannot open that path.

Current guidance:

- For code scenarios that must read a tool-produced **host file**, use
  `--code-backend local`. Local execution runs on the host, so the returned path
  resolves directly.
- Use the `docker` backend for computation that does not need to open a
  tool-produced host file (the data the code needs is in the conversation), where
  isolation matters more than direct file access.

> Mounting a host directory into the Docker sandbox is not currently supported by
> Stirrup's stock `DockerCodeExecToolProvider`. If that lands upstream
> (an `extra_mounts` option), the Docker backend can read host files directly.

---

## CLI flags

In addition to the [common flags](../INSTRUCTIONS.md#common-flags) (`--model-id`,
`--show-trajectory`, `--json`, `--run-id`, `--scenario-id`, ...):

| Flag                  | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| `--code-enabled`      | Enable code execution (default). The code track.                                     |
| `--no-code`           | Tools-only; comparable to the other runners.                                         |
| `--code-backend`      | `docker` (default), `local`, or `e2b`.                                               |
| `--max-turns N`       | Max agent turns (default: 30).                                                       |
| `--max-tokens N`      | Max output tokens per model call; keep under the provider limit (default: 16384).    |

Environment variable: `STIRRUP_CODE_IMAGE` (Docker image; default `python:3.12-slim`).

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
| `WatsonxException ... number of new tokens 200000 ... exceeds the limit of 100000` | `--max-tokens` too high; keep under 100k (default 16384). |
| `docker.errors.DockerException: Error while fetching server API version ... FileNotFoundError` | SDK can't find the daemon socket. `export DOCKER_HOST=unix://<path from 'docker context inspect | grep Host'>`. |
| Docker connects but `code_exec` hits `ModuleNotFoundError` | Sandbox image lacks the library; build/point `STIRRUP_CODE_IMAGE` at `assetops-code` (or an image with the stack). |
| Agent's code can't open a tool-produced file path in Docker | Expected — the host path isn't in the sandbox. Use `--code-backend local` for file-reading scenarios. |
| `uv sync` fails to resolve | A pin (e.g. `litellm==...`) clashing with Stirrup's range; relax to a compatible range. |
| `stirrup` import errors | `stirrup[mcp,litellm,docker]` not installed; re-run `uv sync`. |
| A server missing from the tool list | Its subprocess failed to start (missing CouchDB creds for `iot`/`wo`/`vibration`, or model load for `tsfm`). |

---

## What was added

- `src/agent/stirrup_agent/` — `runner.py`, `cli.py`, `trajectory.py`, `__init__.py`,
  `Dockerfile.code`, and `tests/test_runner.py`.
- `pyproject.toml` — `stirrup[mcp,litellm,docker]` dependency and the `stirrup-agent`
  entry point.

No other parts of the repo are modified; the MCP servers are unchanged.