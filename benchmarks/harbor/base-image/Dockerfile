# AssetOpsBench runtime base image.
#
# Built once per repo commit and published. Every Harbor task layers only its
# own scenario folder on top of this, which keeps each task's build context at
# a few KB. That matters: Harbor tags a task image by a SHA-256 content hash of
# the whole environment/ directory, so a fat per-task context defeats the cache
# and rebuilds the world for every scenario.
#
#   bash benchmarks/harbor/scripts/build-runtime-image.sh \
#     -t assetopsbench/runtime:$(git rev-parse --short HEAD) --load
#
# The script builds from a clean one-commit clone of HEAD, so `COPY . .` below
# sees committed files only. A plain `docker build` of the working tree would
# also copy untracked files, which .dockerignore alone cannot anticipate.
FROM python:3.12-slim

# HF_HOME fixes the Hugging Face cache at one path. The default follows $HOME,
# so a container running as anyone but root looks elsewhere and refetches every
# weight this image already carries.
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
# --group otel installs the OTLP exporter and httpx instrumentation. The task
# sets OTEL_TRACES_FILE, which turns on the file exporter; without this group
# it fails at export time on a missing google.protobuf and drops every batch.
RUN uv sync --frozen --group otel --extra tsfm --no-install-project

# Repo, including the shared scenario corpus under
# src/couchdb/scenarios_data/shared (7.7 MB in the open profile). Shared here
# means pulled once, not copied into every task context.
COPY . .
RUN uv sync --frozen --group otel --extra tsfm

# Model weights, from benchmarks/harbor/base-image/models.txt: a committed list
# of public Hub repo ids, exported from the model catalog with
#
#   preload_models.py --print-repos > benchmarks/harbor/base-image/models.txt
#
# The list rather than the catalog, because the catalog is private and a COPY
# would write it into a layer for good. check_models_list.sh fails CI when the
# two drift.
RUN uv run python benchmarks/harbor/scripts/preload_models.py \
        --from-list benchmarks/harbor/base-image/models.txt --download

CMD ["bash", "-lc", "sleep infinity"]
