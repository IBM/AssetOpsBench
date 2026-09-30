# Running the code sandbox

`benchmarks/harbor/overlays/code-sandbox.yaml` gives each trial its own
Docker-in-Docker daemon and runs agent-written code there instead of in the
task container. It is opt-in per run.

## Why you would turn it on

With the default `code_backend=local`, Stirrup runs agent-written code in the
`main` container, which holds `COUCHDB_URL` and the forwarded credentials and
can reach the `couchdb` sidecar by name. The code tool can then query the
database directly and skip the MCP tools the benchmark measures. The system
prompt forbids it; nothing enforces it.

Under the overlay, code containers run inside the trial's own daemon with no
environment and no DNS entry for `couchdb`, so that shortcut has neither a
hostname nor a credential.

## Prerequisites

- Docker that allows `privileged: true`, which the `dind` service needs. Only
  local Docker (Docker Desktop, Rancher Desktop, docker-ce) is tested. Harbor's
  cloud environments either refuse privileged containers or already run the
  Compose stack inside their own dind, so treat them as unsupported.
- The runtime image: `bash benchmarks/harbor/scripts/build-runtime-image.sh`.
- A `.env` with your model credentials.
- Headroom: each trial runs `main`, `couchdb`, `dind` and a loader, so use a
  lower `--n-concurrent` than for the tools-only arm.

## The easy path: run.sh

`benchmarks/harbor/run.sh` does all of it: it builds and saves the code image,
generates the tasks and passes the overlay with the required `--ak` flags.

```bash
bash benchmarks/harbor/run.sh \
  -s /path/to/scenarios_data \
  -l /path/to/leaderboard \
  -n 2 \
  -m "litellm_proxy/azure/gpt-5.6-sol max"
```

## The manual path

**1. Build the code image and save it as a tar.** Each dind starts with an
empty image store; the overlay's `code-image-loader` loads the tar into it
before the agent starts.

```bash
docker build -t assetops-code:dev \
  -f src/agent/stirrup_agent/Dockerfile.code src/agent/stirrup_agent
docker save assetops-code:dev -o ~/assetops-code.tar
export AOB_CODE_TAR=~/assetops-code.tar
```

Rebuild and save again whenever `Dockerfile.code` changes. To pull from a
registry instead, push the image and set `AOB_CODE_IMAGE` to its reference,
leaving `AOB_CODE_TAR` unset.

**2. Generate the tasks.**

```bash
uv run python benchmarks/harbor/adapter/generate_tasks.py --overwrite
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

For a private profile, also pass `overlays/private-data.yaml` (see
[README.md](README.md#running-a-private-profile-by-hand)).

## The four flags

- `code_enabled=true` turns the code tool on.
- `code_backend=docker` sends code to the daemon. Without it the overlay starts
  and changes nothing.
- `allow_docker_backend=true` is a guard: `StirrupAgent` refuses
  `code_backend=docker` without it, since a plain task container has no daemon.
- `workspace_dir=/workspace-share` is required too. Stirrup creates its
  workspace on `main` and bind-mounts that path into the code container, but
  with `DOCKER_HOST` pointing at dind the daemon resolves the path inside dind.
  The overlay mounts one volume at `/workspace-share` in both services so the
  path means the same thing on each side. Without it, every spilled MCP result
  goes missing with no error.

## Checking it worked

During a run:

```bash
docker ps --filter name=dind            # one dind per running trial
docker compose -p <project> exec dind docker ps -a
```

Each trial's Compose project (`<task>__<uuid7>__env`) has its own daemon, layer
store and workspace, all destroyed with the trial.

## What this protects, and what it does not

Code containers get no environment: Stirrup injects only the variables in its
`env_vars` list, and `.dockerignore` keeps `.env` out of the image. Code also
cannot read the `main` filesystem; only `/workspace-share` crosses.

It is containment, not a firewall. dind sits on the trial network so `main`
can reach it at `tcp://dind:2375`, and the inner bridge NATs outbound through
it, so a code container that guessed CouchDB's IP could still reach port 5984.
Closing that needs the inner containers on `--network none`, which is
Stirrup's call.

It constrains agent-written code only. The MCP servers still run in `main` and
hold `COUCHDB_URL`; that is the intended path.

## Troubleshooting

**dind will not start.** Almost always `privileged: true` being refused.

**`no AOB_CODE_TAR; the daemon will pull ...` in the loader log.** Expected on
the registry route. Otherwise the `AOB_CODE_TAR` path is wrong or the file is
empty. Under `run.sh` the job's tar path is in `<job>.code-tar` beside the job.

**`pull access denied for assetops-code` on the first `code_exec`.** Neither a
tar nor a pullable `AOB_CODE_IMAGE` was given.

**Artifacts missing, no error.** `workspace_dir` is not `/workspace-share`.
