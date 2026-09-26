# AssetOpsBench on Harbor

A working prototype of one AssetOpsBench scenario as a [Harbor](https://github.com/harbor-framework/harbor) task, plus the agent adapter and the generator that produces the rest. Written against AssetOpsBench `main` at `81265cb`.

The point of the port: Harbor names one Docker Compose project per trial and derives every container, network and volume name from it, so each scenario gets its own CouchDB and concurrent scenarios cannot reach each other's data. That replaces the shared-CouchDB reset in `src/benchmark/scenario_suite_runner.py`, which silently deletes a running scenario's databases the moment the loop is parallelized.

## Layout

```
base-image/Dockerfile                 Shared runtime image, built once per repo commit
agent/assetops_harbor/stirrup.py      Stirrup as a Harbor BaseInstalledAgent
benchmarks/harbor/adapter/generate_tasks.py             Scenario folders -> Harbor task directories
benchmarks/harbor/datasets/assetopsbench-open/
  dataset.toml                        Harbor dataset manifest
  metric.py                           Category rollups
  wosr-1/                             One task = one scenario
    task.toml                         Timeouts, env, healthcheck, resources
    instruction.md                    The scenario question
    environment/Dockerfile            Thin layer over the runtime base
    environment/docker-compose.yaml   CouchDB sidecar
    environment/scenario_1/           Per-scenario data, the build context
    tests/test.sh                     Verifier
    tests/to_reward.py                EvalReport -> Harbor reward.json
    tests/scenarios/scenario_1/       Ground truth for the verifier
    solution/solve.sh                 Oracle
```

## Running it

```bash
# 1. Build and push the runtime base (once per AssetOpsBench commit)
docker build -t icr.io/assetopsbench/runtime:dev \
  -f base-image/Dockerfile <path-to-AssetOpsBench>

# 2. Generate the rest of the open profile from the template
python benchmarks/harbor/adapter/generate_tasks.py \
  --scenario-root <path-to-AssetOpsBench>/src/couchdb/scenarios_data \
  --profile <path-to-AssetOpsBench>/benchmarks/scenario_suite/open.yaml \
  --template benchmarks/harbor/datasets/assetopsbench-open/wosr-1 \
  --output-dir benchmarks/harbor/datasets/assetopsbench-open

# 3. Prove the task is scorable before spending tokens on an agent.
# The open-profile scenarios score with static_json, so no judge model is
# needed. Export AOB_JUDGE_MODEL only for scenarios whose scenario_meta.json
# asks for llm_judge.
harbor run -p benchmarks/harbor/datasets/assetopsbench-open --agent oracle

# 4. Run Stirrup, all scenarios in parallel
PYTHONPATH=agent harbor run \
  -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent assetops_harbor.stirrup:StirrupAgent \
  --model watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
  --ak code_enabled=false \
  --n-concurrent 16
```

## Why the MCP servers need an explicit env

`StirrupAgentRunner._build_mcp_config` sets `env` on every stdio server. It has
to. `mcp.client.stdio` applies `get_default_environment()` when
`StdioServerParameters.env` is None, and that inherits only HOME, LOGNAME, PATH,
SHELL, TERM and USER. Without it, no server sees `COUCHDB_URL` and each falls
back to `http://localhost:5984`.

That default is correct on a laptop, where the shared CouchDB publishes 5984 and
the agent runs on the host, which is why this never surfaced before. Under
Harbor the database is `couchdb:5984` inside the trial's Compose network, the
fallback points at nothing, and the failure appears inside a tool call as

```
Error executing tool list_workorders: All connection attempts failed
```

rather than at startup. `src/agent/tests/test_stirrup_mcp_env.py` guards both
the runner's behaviour and the SDK default it compensates for.

## Credentials

The agent forwards credentials from the Harbor process into the container for
the agent phase only. Exporting them in your shell is enough:

```bash
export TOKENROUTER_BASE_URL=... TOKENROUTER_API_KEY=...
harbor run -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent assetops_harbor.stirrup:StirrupAgent --model tokenrouter/MiniMax-M3
```

`--ae KEY=VALUE` overrides the shell for one run. The forwarded set is
`CREDENTIAL_ENV_VARS` in `src/assetops_harbor/stirrup.py`: the LiteLLM and
TokenRouter router pairs, the watsonx variables, and the common OpenAI,
Anthropic, AWS and Gemini names.

A model with a `litellm_proxy/` or `tokenrouter/` prefix fails at construction
when its pair is unset, before Harbor builds anything. `src/llm/routers.py`
otherwise raises the same thing inside the container, which costs an image build
and a container per trial to learn that a variable is missing.

Note what is NOT used: the repository's `.env`. `.dockerignore` keeps it out of
the image deliberately, so credentials never get baked into a layer that could
be pushed to a registry.

## Arms

Main has no `--topology` flag. The arms `stirrup-agent` actually exposes map onto Harbor agent kwargs, and Harbor records each one in the trial's `config.json`, so the arm shows up in the result rather than being inferred from a directory name.

| Harbor kwarg | CLI flag it produces | Default here |
| --- | --- | --- |
| `--ak code_enabled=false` | `--no-code` (tools-only) | `--code-enabled` |
| `--ak code_backend=local` | `--code-backend local` | `local` |
| `--ak max_turns=50` | `--max-turns 50` | `30` |
| `--ak temperature=0.2` | `--temperature 0.2` | omitted |
| `--ak reasoning_effort=high` | `--reasoning-effort high` | omitted |

`code_backend` defaults to `local` rather than main's `docker`. The docker backend spawns a sibling container from `STIRRUP_CODE_IMAGE`, and a Harbor task container has no Docker daemon, so main's default would fail every run. The adapter rejects `docker` with that explanation unless you pass `allow_docker_backend=true` and wire a socket into the task's compose file. `local` is the right backend under Harbor anyway: the container is already a per-trial sandbox, so the isolation the docker backend buys on a laptop is redundant here.

## Three things main forced

The question reaches the CLI as a file, not on the command line. `stirrup-agent` takes the question as a required positional and has no stdin path, so the adapter uploads the instruction to `/tmp/aob_instruction.txt` and passes `"$(cat ...)"`. That survives multi-line prose without the quoting hazards of inlining it, and it needs no change to `src/agent/cli.py`.

`convert_trajectory` handles both shapes `persistence._serialize_trajectory` emits. The SDK runners produce a dict with `turns` (`TurnRecord`: `index`, `text`, `tool_calls`, `input_tokens`, `output_tokens`, `duration_ms`); plan-execute produces a list of `StepResult` (`step_number`, `task`, `server`, `response`, `error`, `tool`, `tool_args`), which carries no token counts, so `FinalMetrics` stays zero on that path.

`ToolCall.id` defaults to `""` on main, so the converter synthesizes a stable id per call. ATIF rejects a trajectory whose observation results do not match a `tool_call_id` in the same step, so an empty id would fail validation.

## Two Harbor behaviors worth knowing before editing these files

Harbor injects task env, bind mounts and resource limits into the `main` service only. The CouchDB sidecar receives nothing from Harbor, which is why it carries its own credentials and `mem_limit` in the compose file.

Harbor namespaces containers, networks and volumes per trial, but not host ports. Never add a `ports:` block to a task's compose file: it publishes a fixed host port and collides the moment two trials of that task run at once, which is the exact failure this port exists to remove. Use `expose:` and reach services by name.

## Where run output goes

Harbor writes every run to `./jobs/<timestamp>/<trial>/` relative to the working
directory: config, lock, result, the agent's trajectories, rewards and verifier
logs. That directory is gitignored, because it is run output rather than source
and it grows with every invocation.

To keep runs entirely outside the repository, pass `-o`:

```bash
harbor run -p benchmarks/harbor/datasets/assetopsbench-open --agent oracle \
  -o ~/Documents/AssetOpsBenchRuns/harbor
```

If `jobs/` was staged before this entry existed, untrack it without deleting the
files:

```bash
git rm -r --cached jobs
```

## Resource limits

The task pins no `cpus`. Harbor renders that field into both a Docker limit and
a reservation on `main`, so a value above what the Docker daemon is allocated
fails the run before the agent starts:

```
Error response from daemon: range of CPUs is from 0.01 to 2.00, as there are only 2 CPUs available
```

Docker Desktop ships with a low CPU allocation by default, and two concurrent
trials reserve the value twice. Leaving it unpinned lets the daemon share what
it has. To impose limits on a bigger machine, pass them at run time rather than
editing the task:

```bash
harbor run -p benchmarks/harbor/datasets/assetopsbench-open --agent oracle \
  --override-cpus 4 --override-memory-mb 8192
```

`--cpus ignore` and `--memory ignore` drop enforcement entirely, which is the
quickest way past a resource error on a constrained machine.

## Scoring

`evaluation.loader.load_scenario_dirs` sets `scoring_method` per scenario, from `scenario_meta.json` when that file is present and `static_json` otherwise, and `evaluator._score_one` prefers it over `--scorer-default`. All three open-profile scenarios therefore score with `static_json` and need no judge model. `tests/test.sh` passes `--judge-model` only when `AOB_JUDGE_MODEL` is set, because passing it empty fails, and omitting it fails any scenario that does ask for `llm_judge`.

`tests/test.sh` passes `--scorer-default static_json` explicitly. The CLI's own
default is `llm_judge`, but `evaluation.cli` registers that scorer only inside
`_maybe_install_judge()`, which no-ops without `--judge-model`. It then calls
`_validate_scorer_default()` and exits with

```
unknown scorer 'llm_judge'; registered: ['fmea', 'static_json']
```

before scoring anything, which surfaces as a reward of 0 and no exception. This
is only the fallback: the scenario's own `scoring_method` still wins.

The adapter copies every file that loader reads into `tests/scenarios/scenario_N/`: `question.txt`, `groundtruth.txt`, `groundtruth_eval.json`, `scenario_meta.json`, `rubric.json`, `reference_answer.json`. Copying only the first two would silently downgrade an `llm_judge` scenario to `static_json`.

`tests/test.sh` uses neither `set -e` nor `${VAR:?}`. A missing `reward.json` raises `RewardFileNotFoundError`, which Harbor classes as a harness failure rather than a score of zero and puts on its non-retryable list, so every exit path has to reach `to_reward.py`.

## What has been verified, and how

Run against Harbor installed from source and AssetOpsBench `main` at `81265cb`:

- `task.toml` validates against Harbor's `TaskConfig` for all three generated tasks.
- Every `${VAR:-}` template in `[environment].env` and `[verifier].env` resolves with nothing exported.
- Harbor's `AgentFactory` resolves `assetops_harbor.stirrup:StirrupAgent` and constructs it from `--ak` string values; the docker-backend guard fires and `allow_docker_backend=true` overrides it.
- `convert_trajectory` produces trajectories that pass strict ATIF validation for all three shapes `persistence._serialize_trajectory` emits, built from the real `agent.models` and `plan_execute.models` dataclasses.
- Harbor's `load_model_usage_from_trajectory` reads 2600 input and 92 output tokens back out of that ATIF, so token accounting works.
- The oracle scores **1.0 on all three scenarios** through AssetOpsBench's real `Evaluator`, and `to_reward.py` turns each report into `{"reward": 1.0, "passed": 1}`, which `VerifierResult` accepts.
- `docker compose config` on Harbor's own `-f` stacking order yields, per project: no host ports, `expose: 5984` only, `COUCHDB_URL` on `main` and not on the sidecar, `depends_on` wired, and distinct project names for two concurrent trials.
- `evaluate`'s parser accepts exactly the flags `tests/test.sh` passes; the loader reads the generated `tests/scenarios` layout.

## Still unverified

Two things need a machine with a Docker daemon.

The base image build. `uv sync --frozen` against the repo has not been exercised in a slim Python image here, and the Stirrup code track pulls a sandbox dependency set on top. Step 3 surfaces any image problem before tokens are spent.

Live two-container isolation. The compose merge is verified but no containers were started. Confirm it once:

```bash
harbor run -p benchmarks/harbor/datasets/assetopsbench-open --agent oracle --n-concurrent 2 &
sleep 20 && docker ps --format '{{.Names}}\t{{.Ports}}' | grep couchdb
```

Two rows, different project prefixes, no host ports. One row means the trials serialized; a host port means someone added a `ports:` block.
