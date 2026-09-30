# Running AssetOpsBench on Harbor

Runs the benchmark's scenarios in parallel, each with its own CouchDB, using
[Harbor](https://github.com/harbor-framework/harbor). Start to finish in about
ten minutes, most of it waiting on an image pull.

`README.md` in this directory explains how it works and why. This file is just
the steps.

## 1. Prerequisites

- Docker, running. Give it at least 4 GB of memory in Docker Desktop settings.
- [uv](https://docs.astral.sh/uv/)
- An API key for whichever model you want to evaluate.

```bash
git clone https://github.com/IBM/AssetOpsBench.git
cd AssetOpsBench
uv sync --dev --extra harbor
```

`--extra harbor` matters. Harbor is optional and a plain `uv sync` leaves it out.

## 2. Get the runtime image

Every task image layers its scenario onto one shared runtime image. Pull the
published one (currently linux/arm64 only) and point the tasks at it:

```bash
docker pull quay.io/assetopsbench/runtime:dev
export AOB_RUNTIME_IMAGE=quay.io/assetopsbench/runtime:dev
```

Or build it yourself, which takes a few minutes and needs no registry:

```bash
bash benchmarks/harbor/scripts/build-runtime-image.sh
```

The script builds `assetopsbench/runtime:dev`, also tagged
`assetopsbench/runtime:<commit>`, from `git archive HEAD`, not from your
working tree, so untracked files such as local results never reach
the container the agent runs in. Commit a change first to include it.

Each task's `environment/docker-compose.yaml` passes `AOB_RUNTIME_IMAGE` to
its Dockerfile as a build arg, so `harbor run` builds FROM whatever that
variable names when it runs, and from the local `assetopsbench/runtime:dev`
when it is unset. Changing the image needs no task regeneration. `run.sh` takes
it as `-r`.

## 3. Generate the tasks

```bash
uv run python benchmarks/harbor/adapter/generate_tasks.py --overwrite
```

One Harbor task per scenario appears under
`benchmarks/harbor/datasets/assetopsbench-open/`. They are gitignored and you
can regenerate them at any time.

## 4. Check it works, before spending tokens

```bash
uv run harbor run -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent oracle --n-concurrent 2
```

Expect 3 trials, 0 exceptions, Passed 1.000, Reward 1.000. The oracle writes the
known answer, so a perfect score means the environment, the data load and the
scoring all work. Anything less is a setup problem, not a model problem.

While it runs, in another shell:

```bash
docker ps --format '{{.Names}}\t{{.Ports}}' | grep couchdb
```

Two CouchDB containers under different project prefixes, neither publishing a
host port. That is the per-trial isolation, and it is why the scenarios can run
at the same time without overwriting each other's data.

## 5. Run an agent

```bash
export TOKENROUTER_BASE_URL=... TOKENROUTER_API_KEY=...

uv run harbor run -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent assetops_harbor.stirrup:StirrupAgent \
  --model tokenrouter/MiniMax-M3 \
  --ak code_enabled=false \
  --n-concurrent 2 \
  -o ~/AssetOpsBenchRuns/harbor
```

Export the credentials your model needs and the agent forwards them into the
agent phase only. The supported names are in `CREDENTIAL_ENV_VARS` in
`src/assetops_harbor/stirrup.py`: the LiteLLM and TokenRouter pairs, the watsonx
variables, and the usual OpenAI, Anthropic, AWS and Gemini names. The agent
checks for the pair its model needs at construction time, so a missing key fails
the job immediately instead of after three trials.

`--ae KEY=VALUE` overrides the shell for a single run. `code_enabled=false` runs
the tools-only track, which is the arm comparable to the other runners.

## 6. Read the results

```
<jobs-dir>/<timestamp>/<task>__<uuid>/
├── result.json                  rewards, token totals, the agent config used
├── agent/trajectory.json        the run in ATIF form
├── agent/traces.jsonl           OTLP spans
└── verifier/reward.json         the score, and the evaluation logs beside it
```

```bash
uv run harbor view ~/AssetOpsBenchRuns/harbor
```

## The code track

Agent-written code runs in the same container as the MCP servers by default,
which means it can query CouchDB directly and bypass the tools the benchmark
measures. `benchmarks/harbor/overlays/code-sandbox.yaml` moves it into a
per-trial Docker-in-Docker daemon:

```bash
uv run harbor run -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent assetops_harbor.stirrup:StirrupAgent \
  --model tokenrouter/MiniMax-M3 \
  --extra-docker-compose benchmarks/harbor/overlays/code-sandbox.yaml \
  --ak code_enabled=true --ak code_backend=docker \
  --ak allow_docker_backend=true --ak workspace_dir=/workspace-share \
  --n-concurrent 1
```

`workspace_dir=/workspace-share` is required, not optional. Stirrup creates its
workspace on the main container's filesystem and bind-mounts that path into the
code container, so with `DOCKER_HOST` pointing at the daemon the path has to
exist on both sides. The overlay's shared volume is what makes it resolve.

Code containers get no DNS entry for `couchdb` and no environment at all, so the
realistic shortcut has neither a hostname nor a credential. Read the README
section "The code track and CouchDB" for what this does and does not guarantee.
Lower `--n-concurrent` for this arm: each trial now runs a Docker daemon of its
own.

Supply the sandbox image one of two ways. Point at a published one:

```bash
export AOB_CODE_IMAGE=assetopsbench/code:latest
```

Or hand the daemon a local tar, which needs no registry at all:

```bash
docker save assetops-code:dev -o ~/assetops-code.tar
export AOB_CODE_TAR=~/assetops-code.tar
```

## Common problems

**`range of CPUs is from 0.01 to 2.00`** — Docker Desktop has fewer CPUs than
the run wants. Raise it in settings, or pass `--cpus ignore`.

**`unknown scorer 'llm_judge'`** — you are calling `evaluate` by hand without
`--scorer-default`. The task's `tests/test.sh` passes it; copy what it does.

**`All connection attempts failed` inside a tool** — the MCP servers cannot
reach CouchDB. Confirm your checkout includes the `env` entry in
`StirrupAgentRunner._build_mcp_config`, without which the MCP SDK hands each
server a six-variable environment that omits `COUCHDB_URL`.

**Trials fail instantly with a pull error** — the runtime image is missing, or
it is tagged under a name the task Dockerfile does not reference. Redo step 2
and check `docker images assetopsbench/runtime`.

**`No module named 'google.protobuf'` during a run** — the image was built
without the `otel` dependency group, which the file trace exporter needs.
Rebuild with `benchmarks/harbor/scripts/build-runtime-image.sh`, whose
Dockerfile passes it.

**The verifier scores 0 but the agent clearly answered** — check
`verifier/test-stderr.txt`. The evaluator joins records to scenarios on
`scenario_id`, so a record carrying the wrong id yields no pairs and no score.
