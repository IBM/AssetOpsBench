# `cli_agent` — benchmarking terminal coding agents

This package lets AssetOpsBench benchmark **CLI coding agents** (Codex, Claude
Code, Gemini CLI, and more) alongside the existing SDK runners. It ports the
per-agent adapter pattern from
[`rdi-berkeley/agents-last-exam`](https://github.com/rdi-berkeley/agents-last-exam)
(`ale_run/agents/<agent>/{config,deployer}.py`) onto this repo's
`AgentRunner` / `Trajectory` / `AgentResult` contract.

## Why this shape

ALE wires every agent to its own computer-use MCP bridges (`cua_mcp_server`,
`vm_mcp_server`) and grades on files the agent writes into a provisioned OS-sandbox
VM. AssetOpsBench is tool-centric instead: an agent answers a question by calling
the six domain MCP servers (`iot`, `utilities`, `fmsr`, `tsfm`, `wo`,
`vibration`) and we grade the trajectory. So we kept ALE's three-method agent
lifecycle and replaced its VM/bridge environment with our MCP servers.

Each agent differs in exactly three places — the same split as ALE's
`config.py` / `deployer.py`:

| Hook | ALE analogue | What it does |
|---|---|---|
| `_write_config(home, base_url)` | `config.py` + `deployer.install()` | Write the agent's MCP + provider config; return extra env |
| `_build_command(home, system_prompt, question)` | `deployer.launch()` | The headless launch argv |
| `_handle_event(event, trajectory)` | `deployer.parse_artifacts()` | Fold one JSON event into the `Trajectory` |

Everything else (env validation, subprocess streaming, timeout, the
observability span, `persist_trajectory`) lives once in `base.py`.

## Layout

Each agent is its own subpackage with `runner.py` + `cli.py`, matching the
existing `openai_agent` / `deep_agent` layout (shared base + providers live at
the top):

```
src/agent/cli_agent/
  __init__.py            # exports + CLI_AGENT_RUNNERS registry
  __main__.py            # convenience: `python -m agent.cli_agent <agent> ...`
  _providers.py          # provider routing (litellm / openrouter / tokenrouter / direct)
  base.py                # CliCodingAgentRunner (the shared 90%)
  codex/
    __init__.py
    runner.py            # CodexCliRunner    — config.toml + codex exec --json
    cli.py               # codex-agent
  claude_code/
    __init__.py
    runner.py            # ClaudeCodeRunner  — .mcp.json + stream-json
    cli.py               # claude-code-agent
  gemini/
    __init__.py
    runner.py            # GeminiCliRunner   — .gemini/settings.json + --output-format json
    cli.py               # gemini-agent
  README.md
```

Each `cli.py` follows the repo convention: `add_common_args` + `print_result`
+ `run_sdk_cli` from `agent._cli_common`, with a `main()` wired as a console
script (so you get `.env` loading, OTEL tracing, `--show-trajectory`, `--json`,
`--verbose` for free, exactly like `openai-agent` / `deep-agent`).

## Install & env

The CLIs are assumed to be on `PATH` (pin versions for reproducibility, as ALE
does — e.g. `@openai/codex@0.114.0`):

```bash
npm i -g @openai/codex @anthropic-ai/claude-code @google/gemini-cli
```

Then set the env for whichever provider(s) you route through (see **Providers**):

```bash
# LiteLLM
export LITELLM_BASE_URL="https://your-litellm-host"   LITELLM_API_KEY="sk-litellm-..."
# OpenRouter (base URL defaults to https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY="sk-or-..."
# TokenRouter (self-hosted, OpenAI-compatible /v1)
export TOKENROUTER_BASE_URL="https://your-tokenrouter/v1"   TOKENROUTER_API_KEY="..."
```

## Providers

Routing is selected by a **prefix on the model id**, generalising the repo's
existing `litellm_proxy/` convention (see `_providers.py`):

| Prefix | Provider | Base URL | API key env |
|---|---|---|---|
| `litellm_proxy/<provider>/<model>` | LiteLLM | `LITELLM_BASE_URL` | `LITELLM_API_KEY` |
| `openrouter/<vendor>/<model>` | OpenRouter | `OPENROUTER_BASE_URL` (default `…/api/v1`) | `OPENROUTER_API_KEY` |
| `tokenrouter/<model>` | TokenRouter | `TOKENROUTER_BASE_URL` | `TOKENROUTER_API_KEY` |
| *(no prefix)* | direct | — (CLI native) | CLI native (e.g. `OPENAI_API_KEY`) |

All three proxies are OpenAI-compatible, but each CLI speaks one native API
flavor, so not every agent works with every provider. Each runner declares a
`supported_providers` set and raises a clear error on a bad combo:

| Agent | API flavor | Providers |
|---|---|---|
| `CodexCliRunner` | OpenAI | litellm, openrouter, tokenrouter, direct |
| `ClaudeCodeRunner` | Anthropic | litellm (`/anthropic`), tokenrouter, direct |
| `GeminiCliRunner` | Gemini | litellm (`/gemini`), tokenrouter, direct |

OpenRouter is OpenAI-format only, so it's excluded from the Anthropic/Gemini
agents. The headline win is that **any OpenAI-flavor agent (Codex) is uniform
across LiteLLM / OpenRouter / TokenRouter** via a single `[model_providers.*]`
block — switching providers is just changing the model prefix.

```python
CodexCliRunner(model="litellm_proxy/azure/gpt-5.4")   # via LiteLLM
CodexCliRunner(model="openrouter/openai/gpt-5.4")     # via OpenRouter
CodexCliRunner(model="tokenrouter/gpt-5.4")           # via TokenRouter
```

## Run — CLI

After wiring the console scripts (below), each agent has its own command,
shaped exactly like `openai-agent` / `deep-agent`:

```bash
codex-agent          --model-id tokenrouter/gpt-5 "What sensors are on Chiller 6?"
claude-code-agent    --model-id litellm_proxy/aws/claude-opus-4-6 --show-trajectory "..."
gemini-agent         --model-id gemini-2.5-pro --json "What is the current time?"
```

Or without installing the scripts, via `uv run` from the repo root:

```bash
uv run python -m agent.cli_agent codex --model-id tokenrouter/gpt-5 "What sensors are on Chiller 6?"
uv run python -m agent.cli_agent --list
```

Common flags (from `_cli_common`): `--model-id`, `--show-trajectory`, `--json`,
`--verbose`, `--run-id`, `--scenario-id`, plus `--timeout`. An empty answer
usually means the CLI's event schema drifted — rerun with `--verbose` and adjust
the agent's `_handle_event`.

## Quickstart — TokenRouter + Codex

TokenRouter is OpenAI-compatible, so Codex is the clean working path:

```bash
npm i -g @openai/codex@0.114.0          # pin for reproducibility
export TOKENROUTER_BASE_URL="https://<your-tokenrouter>/v1"
export TOKENROUTER_API_KEY="<your-token>"

codex-agent --model-id tokenrouter/<model> "What sensors are on Chiller 6?"
# or: uv run python -m agent.cli_agent codex --model-id tokenrouter/<model> "..."
```

The model after the `tokenrouter/` prefix is whatever your TokenRouter exposes
(e.g. `tokenrouter/gpt-5`). Codex writes a `[model_providers.tokenrouter]` block
pointing at `TOKENROUTER_BASE_URL` with `env_key = "TOKENROUTER_API_KEY"`.

## Run — Python

```python
import anyio
from agent.cli_agent import CodexCliRunner

runner = CodexCliRunner(model="tokenrouter/gpt-5")
result = anyio.run(runner.run, "What sensors are on Chiller 6?")
print(result.answer)
```

## Register in the benchmark

1. Export from `src/agent/__init__.py`:

   ```python
   from .cli_agent import CodexCliRunner, ClaudeCodeRunner, GeminiCliRunner
   # add the three names to __all__
   ```

2. Wire the console scripts in `pyproject.toml` (alongside `openai-agent` /
   `deep-agent`):

   ```toml
   [project.scripts]
   codex-agent       = "agent.cli_agent.codex.cli:main"
   claude-code-agent = "agent.cli_agent.claude_code.cli:main"
   gemini-agent      = "agent.cli_agent.gemini.cli:main"
   ```

3. (If your eval harness maps `--runner <name>` to a class) merge in
   `CLI_AGENT_RUNNERS`. Names: `codex`, `claude-code`, `gemini-cli`.

## Adding another agent (Cursor, Grok, Droid, OpenHands, …)

Subclass `CliCodingAgentRunner`, set `agent_name` + `default_model`, and
implement the three hooks. The fastest path is to **read the matching ALE
deployer** (`ale_run/agents/<agent>/deployer.py`, Apache-2.0) for the exact
launch flags, output schema, and version pins, then translate:

- its config writer → `_write_config` (point MCP entries at our `uv run`
  servers instead of the CUA/VM bridge)
- its launch command → `_build_command`
- its transcript parser → `_handle_event` (emit into our `Trajectory`)

## Per-agent caveats to verify

These are the version-sensitive seams — confirm each against the installed CLI:

- **Codex** — `codex exec --json` event schema is experimental. Sanity-check with
  `codex exec --json "hi" | jq -c .` and adjust the item-type names in
  `_handle_event` if a release renamed them.
- **Claude Code** — routes through `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`.
  Your LiteLLM proxy must expose the **Anthropic Messages API** for the model
  (e.g. the `/anthropic` passthrough). `stream-json` requires `--verbose`.
- **Gemini CLI** — does **not** speak the OpenAI wire format. It's pointed at a
  Gemini-compatible endpoint via `GOOGLE_GEMINI_BASE_URL` (LiteLLM's `/gemini`
  passthrough). Confirm the env-var name and passthrough path for your versions;
  this is the likeliest thing to need tweaking.

## Fairness note

These agents are *coding* agents — they also have shell/file tools beyond the
six MCP servers. For an apples-to-apples comparison with the SDK runners,
consider running them inside a container (see the Docker discussion) and/or
restricting their built-in toolset so the only "actions" are MCP tool calls.
