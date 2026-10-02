# Thin per-task layer over the runtime image; the build context is one small
# scenario folder, so Harbor reuses the image across the task's trials.
ARG AOB_RUNTIME_IMAGE=assetopsbench/runtime:dev
FROM ${AOB_RUNTIME_IMAGE}

# Only what init_data.py reads (manifest.json and the files it names), never
# the answers: this image is the agent's container. For an external
# --scenario-root the generator retargets this COPY to /opt/suite/scenarios_data.
COPY scenario_1/ /opt/aob/src/couchdb/scenarios_data/scenario_1/
