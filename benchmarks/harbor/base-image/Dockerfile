# AssetOpsBench runtime base image.
#
# Built once per repo commit and published. Every Harbor task layers only its
# own scenario folder on top of this, which keeps each task's build context at
# a few KB. That matters: Harbor tags a task image by a SHA-256 content hash of
# the whole environment/ directory, so a fat per-task context defeats the cache
# and rebuilds the world for every scenario.
#
#   docker build -t icr.io/assetopsbench/runtime:$(git rev-parse --short HEAD) \
#     -f base-image/Dockerfile <path-to-AssetOpsBench>
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    AOB_HOME=/opt/aob

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
RUN uv sync --frozen --group otel --no-install-project

# Repo, including the shared scenario corpus under
# src/couchdb/scenarios_data/shared (7.7 MB in the open profile). Shared here
# means pulled once, not copied into every task context.
COPY . .
RUN uv sync --frozen --group otel

CMD ["bash", "-lc", "sleep infinity"]
