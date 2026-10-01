# AssetOpsBench runtime base image. Every Harbor task layers only its own
# scenario folder on top, keeping each task's build context small.
#
#   bash benchmarks/harbor/scripts/build-runtime-image.sh
#
# The script builds from `git archive HEAD` and passes the commit as
# AOB_COMMIT; a plain `docker build` of the working tree fails the `commit`
# stage, since it would copy untracked files too.

# A separate stage, so AOB_COMMIT (new every commit) does not bust the cache of
# the stage below.
FROM python:3.12-slim AS commit
ARG AOB_COMMIT
RUN test -n "$AOB_COMMIT" || { \
        echo "build with benchmarks/harbor/scripts/build-runtime-image.sh," \
             "not docker build: it builds HEAD, never the working tree" >&2; \
        exit 1; } \
    && echo "$AOB_COMMIT" > /aob-commit

FROM python:3.12-slim

# HF_HOME pins the Hugging Face cache, so a non-root user finds the preloaded
# weights.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    AOB_HOME=/opt/aob \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /opt/aob

# Lockfile first so dependency layers survive source edits.
COPY pyproject.toml uv.lock ./
# --group otel: the task sets OTEL_TRACES_FILE, whose exporter needs it.
# --no-dev: the dev group is pytest and the Jupyter stack, which nothing in a
# trial imports.
RUN uv sync --frozen --no-dev --group otel --extra tsfm --no-install-project

# Repo, including src/couchdb/scenarios_data/shared, pulled once rather than
# copied into every task context.
COPY . .
RUN uv sync --frozen --no-dev --group otel --extra tsfm

# The environment is final. Every process in a trial starts through `uv run`
# (healthcheck, agent, MCP servers, verifier), and `uv run` otherwise syncs the
# default groups first, which would put the dev group back, downloading it in
# every trial.
ENV UV_NO_SYNC=1

# Model weights, from base-image/models.txt: public Hub repo ids exported from
# the model catalog, so the build needs no access to a private catalog. Nothing
# keeps the two in step; rerun this whenever the catalog's Hub entries change:
#
#   preload_models.py --print-repos > benchmarks/harbor/base-image/models.txt
RUN uv run python benchmarks/harbor/scripts/preload_models.py \
        --from-list benchmarks/harbor/base-image/models.txt --download

# No .git in the image; StirrupAgent.get_version_command reads the commit here.
COPY --from=commit /aob-commit /opt/aob/.aob-commit

CMD ["bash", "-lc", "sleep infinity"]
