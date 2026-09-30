# Thin per-task layer. Everything heavy lives in the base image, so this build
# context is one small scenario folder and Harbor's content-hash cache reuses
# the image across every trial of this task.
ARG AOB_RUNTIME_IMAGE=assetopsbench/runtime:dev
FROM ${AOB_RUNTIME_IMAGE}

# Only what init_data.py reads, manifest.json and the files it names, never the
# answers: this image is the agent's container. Redundant for the open profile,
# whose scenarios are already in the base image, but it is how a private suite's
# scenario reaches its task: for an external --scenario-root the generator
# points this COPY at /opt/suite/scenarios_data instead.
COPY scenario_1/ /opt/aob/src/couchdb/scenarios_data/scenario_1/
