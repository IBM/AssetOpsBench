# OpenCode Agent

The `opencode-agent` runner drives [OpenCode](https://opencode.ai/docs/) against
the AssetOpsBench MCP servers. It is a peer to `plan-execute`, `claude-agent`,
`openai-agent`, `deep-agent`, and `stirrup-agent`: same CLI contract, same
persisted `Trajectory`, scored by the same `uv run evaluate`.

OpenCode is a general CLI coding agent, but this runner is configured as a
**tools-first benchmark agent** by default. In the default mode, it can call the
AssetOpsBench MCP servers while local file reads, shell commands, file edits,
web access, external directory access, and follow-up questions are denied unless
explicitly enabled.

The runner also supports an optional **CLI workspace mode**. In that mode, each
benchmark run gets a dedicated workspace directory and OpenCode can be allowed
to inspect files and/or run shell commands inside that run workspace.

## Contents

- [Why OpenCode](#why-opencode)
- [Install](#install)
- [Quick start](#quick-start)
- [Execution modes](#execution-modes)
- [Model routing](#model-routing)
- [Permissions and web access](#permissions-and-web-access)
- [Headless server mode](#headless-server-mode)
- [CLI flags](#cli-flags)
- [Benchmark-suite usage](#benchmark-suite-usage)
- [Validation runs](#validation-runs)
- [Troubleshooting](#troubleshooting)
- [What was added](#what-was-added)

---

## Why OpenCode

- **CLI-backed agent loop**, so it can be evaluated as an external agent without
  embedding another SDK directly in AssetOpsBench.
- **MCP support**, so it connects to the same six AssetOpsBench servers as every
  other tool-using runner.
- **Headless automation**, via `opencode run --format json`, which makes it
  usable in scripts and benchmark runs.
- **Optional CLI workspace mode**, so benchmark runs can evaluate agents that
  write and run local analysis code.
- **Server/API mode**, via `opencode serve`, which can keep OpenCode warm across
  repeated runs.
- **Provider flexibility**, including direct OpenCode providers and
  AssetOpsBench router prefixes such as `tokenrouter/` and `litellm_proxy/`.

---

## Install

The Python entry point is declared in `pyproject.toml` as the `opencode-agent`
script. The OpenCode executable itself is installed outside the Python package.

Install the project dependencies from the repo root:

```bash
uv sync
```

Confirm the AssetOpsBench entry point resolved:

```bash
uv run opencode-agent --help
```

Confirm the OpenCode binary is available:

```bash
opencode --version
```

If `opencode` is missing, install it with one of the methods from the OpenCode
docs, for example:

```bash
npm install -g opencode-ai
```

Run the unit tests for config, permissions, workspace handling, and trajectory
mapping:

```bash
uv run pytest src/agent/opencode_agent/tests/ -q
```

---

## Quick start

Set the provider credentials for the model route you want, then run a small
question:

```bash
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
export TOKENROUTER_API_KEY=...

uv run opencode-agent --show-trajectory \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"
```

In the `--show-trajectory` output, look for domain tool calls such as
`iot_sites`, `iot_registry_assets`, or `wo_list_workorders`. That confirms
OpenCode discovered and called the AssetOpsBench MCP tools.

> **Quiet runs.** `opencode-agent` runs OpenCode as a subprocess. During long
> questions, the terminal may look quiet until OpenCode finishes. The persisted
> trajectory is written only after the run completes.

---

## Execution modes

### Default MCP-only mode

By default, OpenCode is allowed to use AssetOpsBench MCP tools only.

The following capabilities are denied by default:

- local file inspection
- shell/bash commands
- file edits
- web fetch/search
- external directory access
- follow-up questions

This is the recommended mode for normal benchmark runs where answers should come
from the configured MCP tools.

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3
```

### CLI workspace mode

CLI workspace mode is opt-in. It gives each OpenCode run a dedicated workspace
directory and can allow file inspection and/or shell execution.

This mode is useful when evaluating CLI-style agents that can write small Python
scripts, run local analysis, and use intermediate artifacts while still using
MCP tools for operational data.

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
  --opencode-workspace-root traces/opencode_workspaces \
  --opencode-allow-files \
  --opencode-allow-bash
```

For scenario `401`, this creates a workspace such as:

```text
traces/opencode_workspaces/opencode_agent_401
```

OpenCode is instructed to use the current working directory as the run workspace
when file or bash access is enabled. It should write scripts, temporary files,
intermediate data, and final artifacts there.

> **Safety note.** `--opencode-allow-bash` is not a hard OS-level sandbox. A
> shell or Python process can still attempt to access files outside the workspace.
> For strict filesystem isolation, run the benchmark inside Docker or another
> sandboxed environment.

---

## Model routing

The runner's default model is `opencode/gpt-5.1-codex`.

| `--model-id` prefix     | OpenCode provider config                | Required env vars |
| ----------------------- | --------------------------------------- | ----------------- |
| `opencode/<model>`      | OpenCode built-in provider              | OpenCode auth / provider setup |
| `openai/<model>`        | OpenCode built-in OpenAI provider       | `OPENAI_API_KEY` or OpenCode auth |
| `anthropic/<model>`     | OpenCode built-in Anthropic provider    | `ANTHROPIC_API_KEY` or OpenCode auth |
| `tokenrouter/<model>`   | Custom OpenAI-compatible `tokenrouter` provider with explicit model registration | `TOKENROUTER_BASE_URL`, `TOKENROUTER_API_KEY` |
| `litellm_proxy/<model>` | Custom OpenAI-compatible `litellm-proxy` provider with explicit model registration | `LITELLM_BASE_URL`, `LITELLM_API_KEY` |

For router prefixes, the runner generates inline OpenCode config using
`OPENCODE_CONFIG_CONTENT`. For example:

```text
tokenrouter/MiniMax-M3 -> tokenrouter/MiniMax-M3
litellm_proxy/azure/gpt-5.4 -> litellm-proxy/azure/gpt-5.4
```

The custom provider route is important because OpenCode's built-in `openai/*`
provider validates model names against its own model registry. Router-hosted
models such as `MiniMax-M3` must be registered explicitly.

---

## Permissions and web access

The runner configures OpenCode permissions for benchmark use:

| Capability | Default | Flag to allow |
| ---------- | ------- | ------------- |
| AssetOpsBench MCP tools | allowed | always enabled |
| `read`, `glob`, `grep`, `lsp` | denied | `--allow-files` |
| shell commands | denied | `--allow-bash` |
| file edits | denied | `--allow-edit` |
| web fetch/search | denied | `--allow-web` |
| external directory access | denied | not exposed |
| follow-up questions | denied | not exposed |

Benchmark runs should not pass `--allow-web`. Without that flag, OpenCode's
`webfetch` and `websearch` permissions are explicitly set to `deny`.

The CLI auto-approves allowed permissions by default so unattended benchmark
runs do not block on prompts:

```text
--dangerously-skip-permissions
```

This does **not** grant access to explicitly denied tools. For example, web
access remains denied unless `--allow-web` is passed.

If you want OpenCode to ask before using allowed tools during local debugging,
pass:

```bash
--ask-permissions
```

Do not use `--ask-permissions` for batch benchmark runs; it can wait for user
input.

---

## Headless server mode

OpenCode can also run as a long-lived headless server:

```bash
opencode serve --port 4096
```

Then attach benchmark calls to it:

```bash
uv run opencode-agent --attach http://localhost:4096 \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"
```

This can reduce startup overhead when running many scenarios. The default runner
does not require server mode; it starts OpenCode through `opencode run`.

---

## CLI flags

In addition to the [common flags](../INSTRUCTIONS.md#common-flags) (`--model-id`,
`--show-trajectory`, `--json`, `--run-id`, `--scenario-id`, ...):

| Flag | Description |
| ---- | ----------- |
| `--max-steps N` | Maximum OpenCode agentic iterations (default: 30). |
| `--agent-name NAME` | Inline OpenCode agent name (default: `assetops`). |
| `--opencode-bin PATH` | OpenCode executable path (default: `opencode`). |
| `--attach URL` | Attach to a running `opencode serve` instance. |
| `--timeout-s N` | Wall-clock timeout for `opencode run` (default: 900). |
| `--allow-files` | Allow file inspection tools (`read`, `glob`, `grep`, `lsp`). Disabled by default. |
| `--allow-bash` | Allow shell commands. Disabled by default. |
| `--allow-edit` | Allow file edits. Disabled by default. |
| `--allow-web` | Allow web fetch/search. Disabled by default. |
| `--workspace-dir PATH` | Workspace directory required when files, bash, or edits are enabled. |
| `--ask-permissions` | Do not auto-approve allowed permissions. Not suitable for batch runs. |

Direct CLI workspace example:

```bash
uv run opencode-agent \
  --model-id tokenrouter/MiniMax-M3 \
  --workspace-dir /tmp/assetopsbench-opencode/smoke \
  --allow-files \
  --allow-bash \
  "Analyze the maintenance cost distribution."
```

---

## Benchmark-suite usage

The benchmark method name uses an underscore:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3
```

The direct CLI command uses a hyphen:

```bash
uv run opencode-agent --model-id tokenrouter/MiniMax-M3 "What sites are available?"
```

For a faster smoke test, skip evaluation:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
  --no-evaluate
```

To run with CLI workspace capabilities:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
  --opencode-workspace-root traces/opencode_workspaces \
  --opencode-allow-files \
  --opencode-allow-bash
```

Scenario-suite OpenCode flags:

| Flag | Description |
| ---- | ----------- |
| `--opencode-workspace-root PATH` | Root directory for per-run OpenCode workspaces. |
| `--opencode-allow-files` | Pass `--allow-files` to `opencode-agent`. |
| `--opencode-allow-bash` | Pass `--allow-bash` to `opencode-agent`. |
| `--opencode-allow-edit` | Pass `--allow-edit` to `opencode-agent`. |

If any OpenCode file, bash, or edit capability is enabled, then
`--opencode-workspace-root` is required.

---

## Validation runs

A reproducible checklist a teammate can follow to confirm a working setup:

```bash
# 0. install + entry point
uv sync
uv run opencode-agent --help
opencode --version

# 1. unit tests (no model call)
uv run pytest src/agent/opencode_agent/tests/ -q

# 2. direct MCP-only smoke test
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
export TOKENROUTER_API_KEY=...
uv run opencode-agent --show-trajectory \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"

# 3. direct CLI workspace smoke test
uv run opencode-agent --show-trajectory \
  --model-id tokenrouter/MiniMax-M3 \
  --workspace-dir /tmp/assetopsbench-opencode/smoke \
  --allow-files \
  --allow-bash \
  "What is the current time?"

# 4. persist a trajectory
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories
uv run opencode-agent --run-id opencode-smoke --scenario-id smoke \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"

# 5. benchmark-suite dry run
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
  --dry-run

# 6. benchmark-suite CLI workspace dry run
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
  --opencode-workspace-root traces/opencode_workspaces \
  --opencode-allow-files \
  --opencode-allow-bash \
  --dry-run
```

---

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `opencode: command not found` | Install OpenCode (`npm install -g opencode-ai`) or pass `--opencode-bin /path/to/opencode`. |
| `Model not found: openai/MiniMax-M3` | Old router mapping. Use the current runner, which maps `tokenrouter/<model>` to a custom `tokenrouter` provider instead of `openai/<model>`. |
| `TOKENROUTER_BASE_URL and TOKENROUTER_API_KEY must be set` | Set both env vars when using `tokenrouter/*`. The base URL should be OpenAI-compatible, usually ending in `/v1`. |
| `LITELLM_BASE_URL and LITELLM_API_KEY must be set` | Set both env vars when using `litellm_proxy/*`. |
| `--workspace-dir is required when enabling files, edits, or bash` | Direct `opencode-agent` runs require a workspace when `--allow-files`, `--allow-bash`, or `--allow-edit` is passed. |
| `--opencode-workspace-root is required when enabling OpenCode files, bash, or edits` | Scenario-suite runs require a workspace root when any OpenCode CLI capability is enabled. |
| Workspace folders are empty | Normal if OpenCode did not need to write scripts or intermediate files. The workspace is available as scratch space, but files are created only if the model/tooling writes them. |
| Run appears stuck after `Command: uv run opencode-agent ...` | Normal for long scenarios: `scenario_suite_runner` is quiet while OpenCode works. Check `traces/trajectories/scenario_suite/opencode_agent/` after completion. |
| Evaluator fails even though the answer text contains the right label | The answer may be too verbose for `static_json` extraction. Prefer concise answers such as `C` or JSON like `{"answer": "C"}` for strict structured tasks. |
| Web access concern | Web is denied by default. Do not pass `--allow-web` for benchmark runs. |
| Bash isolation concern | `--allow-bash` is not a hard OS sandbox. Use Docker or another sandbox for strict filesystem isolation. |
| OpenCode prompts for permissions in a batch run | Do not pass `--ask-permissions`; the default auto-approves allowed tools and still denies explicitly denied tools. |

---

## What was added

- `src/agent/opencode_agent/` — `runner.py`, `cli.py`, `__init__.py`, and
  `tests/test_runner.py`.
- `pyproject.toml` — `opencode-agent` entry point.
- `src/benchmark/scenario_suite_runner.py` — `opencode_agent` benchmark method
  and optional OpenCode workspace flags.
- `docs/opencode-agent.md` — OpenCode usage, permissions, and workspace-mode
  documentation.

No MCP servers are modified. The runner generates OpenCode MCP/provider config
at runtime.