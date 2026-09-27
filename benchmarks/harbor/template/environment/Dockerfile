# Thin per-task layer. Everything heavy lives in the base image, so this build
# context is a single 16 KB scenario folder and Harbor's content-hash cache
# reuses the image across every trial of this task.
ARG AOB_RUNTIME_IMAGE=assetopsbench/runtime:dev
FROM ${AOB_RUNTIME_IMAGE}

# Redundant for the open profile (these three scenarios ship in the repo, so
# they are already in the base image) but it is the mechanism that lets the
# restricted corpus layer its scenarios onto a public base.
COPY scenario_1/ /opt/aob/src/couchdb/scenarios_data/scenario_1/
