# OpenCode Agent

The `opencode-agent` runner drives [OpenCode](https://opencode.ai/docs/) against
the AssetOpsBench MCP servers. It is a peer to `plan-execute`, `claude-agent`,
`openai-agent`, `deep-agent`, and `stirrup-agent`: same CLI contract, same
persisted `Trajectory`, scored by the same `uv run evaluate`.

OpenCode is a general CLI coding agent, but this runner is configured as a
**tools-first benchmark agent** by default. It can call the AssetOpsBench MCP
servers, read/search local files, and return an answer. Shell commands, file
edits, web access, and follow-up questions are denied unless explicitly enabled.

## Contents

- [Why OpenCode](#why-opencode)
- [Install](#install)
- [Quick start](#quick-start)
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

Confirm the AssetOpsBench entry point resolved (no model call):

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

Run the unit tests for the config and trajectory mapping:

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
| `read`, `glob`, `grep`, `lsp` | allowed | always enabled |
| shell commands | denied | `--allow-bash` |
| file edits | denied | `--allow-edit` |
| web fetch/search | denied | `--allow-web` |
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
| `--allow-bash` | Allow shell commands. Disabled by default. |
| `--allow-edit` | Allow file edits. Disabled by default. |
| `--allow-web` | Allow web fetch/search. Disabled by default. |
| `--ask-permissions` | Do not auto-approve allowed permissions. Not suitable for batch runs. |

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

# 2. direct smoke test
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
export TOKENROUTER_API_KEY=...
uv run opencode-agent --show-trajectory \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"

# 3. persist a trajectory
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories
uv run opencode-agent --run-id opencode-smoke --scenario-id smoke \
  --model-id tokenrouter/MiniMax-M3 \
  "What sites are available?"

# 4. benchmark-suite dry run
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /path/to/scenarios_data \
  --agent_name opencode_agent \
  --model-id tokenrouter/MiniMax-M3 \
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
| Run appears stuck after `Command: uv run opencode-agent ...` | Normal for long scenarios: `scenario_suite_runner` is quiet while OpenCode works. Check `traces/trajectories/scenario_suite/opencode_agent/` after completion. |
| Evaluator fails even though the answer text contains the right label | The answer may be too verbose for `static_json` extraction. Prefer concise answers such as `C` or JSON like `{"answer": "C"}` for strict structured tasks. |
| Web access concern | Web is denied by default. Do not pass `--allow-web` for benchmark runs. |
| OpenCode prompts for permissions in a batch run | Do not pass `--ask-permissions`; the default auto-approves allowed tools and still denies explicitly denied tools. |

---

## What was added

- `src/agent/opencode_agent/` — `runner.py`, `cli.py`, `__init__.py`, and
  `tests/test_runner.py`.
- `pyproject.toml` — `opencode-agent` entry point.
- `src/benchmark/scenario_suite_runner.py` — `opencode_agent` benchmark method.

No MCP servers are modified. The runner generates OpenCode MCP/provider config
at runtime.
