# Running the code sandbox

`benchmarks/harbor/overlays/code-sandbox.yaml` gives each trial its own
Docker-in-Docker daemon and runs agent-written code inside it, instead of in the
task container. Opt in per run; nothing uses it unless you pass the overlay.

Read this alongside the header comment in the overlay itself, which explains the
design. This file is about operating it.

## Why you would turn it on

With the default `code_backend=local`, Stirrup executes agent-written code in
the `main` container. That container holds `COUCHDB_URL` and the credentials,
and sits one DNS hop from the `couchdb` sidecar, so the code tool can query the
database directly and skip the MCP layer, which is the thing the benchmark
measures. The system prompt forbids it. Nothing enforces it.

The overlay moves code execution into a per-trial daemon whose containers get no
environment and no DNS entry for `couchdb`, so the realistic shortcut has
neither a hostname nor a credential to work with.

## Prerequisites

Docker that allows `privileged: true`. The dind service needs it.

Which Harbor environments can run this is narrower than it first looks, and
Harbor gives no direct answer, because `EnvironmentCapabilities` models
`docker_compose` but has no `privileged` flag at all. What the source does
settle:

Fourteen environments declare `docker_compose=True`, so only these can take
`--extra-docker-compose` in the first place: `docker`, `ec2`, `gke`,
`langsmith`, `runta`, `beam`, `blaxel`, `daytona`, `hyperbrowser`, `islo`,
`modal`, `novita`, `tensorlake`, `vercel`.

Eleven of those route the stack through `dind_compose`, meaning your compose
file already runs inside a Docker-in-Docker layer the provider owns: `beam`,
`blaxel`, `daytona`, `ec2`, `gke`, `hyperbrowser`, `islo`, `modal`, `novita`,
`tensorlake`, `vercel`. This overlay then asks for a *second*, privileged dind
nested inside that one, which only works if the outer layer is itself
privileged and willing to pass it down.

In practice:

- **`docker` (local)** is the supported path and what `run.sh` uses. You own the
  daemon, so privileged is yours to grant. Docker Desktop and Rancher Desktop
  both allow it.
- **`ec2`** runs on your own instance, so the same reasoning applies.
- **`gke`** runs its compose stack in a pod it explicitly creates privileged, so
  nesting is plausible here, but nobody has tried this overlay on it.
- **The managed sandbox providers** (`beam`, `blaxel`, `daytona`,
  `hyperbrowser`, `islo`, `modal`, `novita`, `tensorlake`, `vercel`) grant
  privileged at the vendor's discretion, and most do not.

Treat anything other than local Docker as untested for this overlay.

The runtime image:

```bash
docker build -t assetopsbench/runtime:dev -f benchmarks/harbor/base-image/Dockerfile .
```

A `.env` with your model credentials. `harbor run` is invoked through
`uv run --env-file .env`.

Headroom. Each trial now runs `main`, `couchdb`, `dind` and a loader, and each
dind starts with an empty image store. Lower `--n-concurrent` for the code arm
well below what you use for the tools-only arm.

## The easy path: run.sh

`benchmarks/harbor/run.sh` already does all of it. It builds the code image,
saves the tar, exports both variables, generates the dataset and passes the
overlay with the four required `--ak` flags.

```bash
./benchmarks/harbor/run.sh \
  -s /path/to/scenarios_data \
  -l /path/to/leaderboard \
  -n 2 \
  -m "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
```

`-n 2` rather than the default 4, for the reason above. Repeat `-m` to sweep
models. `-p` picks a profile other than `benchmarks/scenario_suite/all.yaml`.

## The manual path

Use this for the open profile or a single ad hoc run.

**1. Build the code image and save it as a tar.** Each dind starts empty, so the
image has to get in. The tar route needs no registry and no login.

```bash
docker build -t assetops-code:dev \
  -f src/agent/stirrup_agent/Dockerfile.code src/agent/stirrup_agent
docker save assetops-code:dev -o ~/assetops-code.tar
export AOB_CODE_TAR=~/assetops-code.tar
```

The `code-image-loader` service loads that tar into the trial's daemon before
the agent starts. It exits 0 either way, so a missing tar is not an error, it
just means Stirrup tries to pull instead.

Registry alternative, if you would rather push:

```bash
export AOB_CODE_IMAGE=assetopsbench/code:dev
```

**2. Generate the dataset.**

```bash
uv run python benchmarks/harbor/adapter/generate_tasks.py
```

**3. Run.**

```bash
uv run --env-file .env harbor run \
  -p benchmarks/harbor/datasets/assetopsbench-open \
  --agent assetops_harbor.stirrup:StirrupAgent \
  --model tokenrouter/MiniMax-M3 \
  --extra-docker-compose benchmarks/harbor/overlays/code-sandbox.yaml \
  --ak code_enabled=true \
  --ak code_backend=docker \
  --ak allow_docker_backend=true \
  --ak workspace_dir=/workspace-share \
  --n-concurrent 2
```

## The four flags, and why none is optional

`code_enabled=true` turns the code tool on at all.

`code_backend=docker` sends code to the daemon rather than running it in `main`.
Without this the overlay starts and changes nothing.

`allow_docker_backend=true` is a deliberate guard. `StirrupAgent` refuses
`code_backend=docker` without it, because that backend spawns a sibling
container and there is no Docker daemon inside a plain Harbor task.

`workspace_dir=/workspace-share` is the one that bites if you forget it, and
`StirrupAgent` refuses without it for good reason. Stirrup creates its workspace
with `mkdtemp` on the `main` filesystem and bind-mounts that path into the code
container. With `DOCKER_HOST` pointing at dind, the daemon resolves the bind
source **inside dind**, so `main` and the code container would look at two
different directories. Every spilled MCP artifact would go missing with no
error. The overlay mounts one named volume at the same absolute path in both
services so the path resolves identically.

## Checking it actually worked

During a run, confirm code containers are being created in the trial's daemon
rather than on your host:

```bash
docker ps --filter name=dind            # one dind per running trial
docker compose -p <project> exec dind docker ps -a
```

Compose namespaces the project per trial (`<task>__<uuid7>__env`), so each
trial has its own daemon, its own layer store and its own workspace. Nothing is
shared across trials and everything is destroyed with the trial.

If code silently produces nothing, check `workspace_dir` first. That is the
failure mode with no error message.

## What this protects, and what it does not

It stops agent-written code reaching CouchDB by name or credential. Code
containers get no environment at all, because Stirrup injects only the variables
in its `env_vars` list and `load_dotenv()` finds nothing, since `.dockerignore`
keeps `.env` out of the image.

Code also cannot read the `main` filesystem. Only `/workspace-share` crosses, so
scenario files sitting in `main`, including `groundtruth.txt`, are out of reach
of the code tool while this overlay is on. That is worth knowing, and it is
exactly what does **not** hold with the default `code_backend=local`, where code
runs in `main` and can read all of it.

It is containment, not a firewall. dind's own `eth0` sits on the trial network
so `main` can reach it at `tcp://dind:2375`, and the inner bridge NATs outbound
through it. A code container that guessed couchdb's IP could still reach port
5984. Closing that needs the inner containers on `--network none`, which is
Stirrup's call rather than the overlay's.

It constrains agent-written code only. The six MCP servers still run in `main`
as stdio children of Stirrup and still hold this trial's `COUCHDB_URL`. That is
the intended path, not a gap.

## Troubleshooting

**dind will not start.** Almost always `privileged: true` being refused. Check
which environment you are on against the list above; local Docker is the only
one this overlay is known to work on.

**"no AOB_CODE_TAR; the daemon will pull ..."** in the loader log. Expected when
you went the registry route. If you meant to use a tar, the path in
`AOB_CODE_TAR` is wrong or the file is empty. It defaults to
`$HOME/assetops-code.tar` under `run.sh`.

**Runs are much slower than the tools-only arm.** Each trial pays for a
container and an image load. Lower `--n-concurrent`.

**Artifacts missing, no error.** `workspace_dir` is not `/workspace-share`.
