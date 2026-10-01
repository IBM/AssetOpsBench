# Running AssetOpsBench on Harbor

Runs the benchmark's scenarios in parallel, each with its own CouchDB, using
[Harbor](https://github.com/harbor-framework/harbor). `README.md` in this
directory explains how it works; this file is just the steps.

## 1. Prerequisites

- Docker, running, with at least 4 GB of memory.
- [uv](https://docs.astral.sh/uv/)
- An API key for whichever model you want to evaluate.

```bash
git clone https://github.com/IBM/AssetOpsBench.git
cd AssetOpsBench
uv sync --dev --extra harbor
```

Harbor is optional, so a plain `uv sync` leaves it out.

## 2. Get the runtime image

Every task image layers its scenario onto one shared runtime image. Pull the
published one (currently linux/arm64 only) and point the tasks at it:

```bash
docker pull quay.io/assetopsbench/runtime:dev
export AOB_RUNTIME_IMAGE=quay.io/assetopsbench/runtime:dev
```

Or build it yourself. The first build downloads the Python dependencies and
about 4 GB of model weights, which takes several minutes; after a source change
it takes seconds:

```bash
bash benchmarks/harbor/scripts/build-runtime-image.sh
```

The script builds `assetopsbench/runtime:dev` (also tagged with the commit)
from `git archive HEAD`, so untracked files and uncommitted changes never reach
the image. Commit a change first to include it.

`harbor run` builds each task FROM `AOB_RUNTIME_IMAGE`, or the local
`assetopsbench/runtime:dev` when it is unset. Set it in every new shell, or put
it in `.env` and run `uv run --env-file .env harbor run ...`.

## 3. Generate the tasks

```bash
uv run python benchmarks/harbor/adapter/generate_tasks.py --overwrite
```

One Harbor task per scenario appears under
`benchmarks/harbor/datasets/assetopsbench-open/`. They are gitignored and can
be regenerated at any time.

## 4. Check it works, before spending tokens

```bash
uv run harbor run -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent oracle --n-concurrent 2
```

Expect 3 trials, 0 exceptions, reward 1.000. The oracle writes the known
answer, so anything less is a setup problem, not a model problem.

While it runs, in another shell:

```bash
docker ps --format '{{.Names}}\t{{.Ports}}' | grep couchdb
```

Two CouchDB containers under different project prefixes, neither publishing a
host port: that is the per-trial isolation.

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

Credentials can also live in the repo's `.env`. The agent forwards them into
the agent phase only (the names are in `CREDENTIAL_ENV_VARS` in
`src/assetops_harbor/stirrup.py`) and fails at once if its model's router pair
is missing. `--ae KEY=VALUE` overrides them for a single run.

`code_enabled=false` runs the tools-only track. To let the agent run code, use
the Docker sandbox in [CODE-SANDBOX.md](CODE-SANDBOX.md).

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

## Common problems

**`range of CPUs is from 0.01 to 2.00`**: Docker has fewer CPUs than the run
wants. Raise it in Docker's settings, or pass `--cpus ignore`.

**Trials fail instantly with a pull error**: the build could not find its
base. `AOB_RUNTIME_IMAGE` is unset in this shell and there is no local
`assetopsbench/runtime:dev`, or it names an image that is not local. Redo
step 2 in this shell.

**`unknown scorer 'llm_judge'`**: you are calling `evaluate` by hand without
`--scorer-default`. Copy what the task's `tests/test.sh` does.

**The verifier scores 0 but the agent clearly answered**: check
`verifier/test-stderr.txt`. The evaluator joins records to scenarios on
`scenario_id`, so a record with the wrong id yields no score.
