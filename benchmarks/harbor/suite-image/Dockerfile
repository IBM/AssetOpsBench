# Runtime image plus an external scenario suite, for running a suite that
# does not ship in the repo (e.g. AssetOpsBenchScenarioGeneration/scenarios_data).
#
# The suite's shared/ data is large (GBs of IoT history) and every scenario's
# manifest resolves against it, so it goes in one layer that all tasks share,
# rather than into each task's build context. The build context is the suite
# folder itself:
#
#   docker build -t assetopsbench/runtime:suite \
#     -f benchmarks/harbor/suite-image/Dockerfile <path-to>/scenarios_data
#
# Generate tasks against it with
#
#   generate_tasks.py --scenario-root <path-to>/scenarios_data \
#     --runtime-image assetopsbench/runtime:suite
#
# An external --scenario-root is what points the data load at
# /opt/suite/scenarios_data (SUITE_DATA_DIR in generate_tasks.py). The suite sits beside the repo copy, not over it: only init_data.py reads it
# (through SCENARIOS_DATA_DIR on the healthcheck), matching
# scenario_suite_runner, which sets that variable for the data load alone.
ARG AOB_RUNTIME_IMAGE=assetopsbench/runtime:dev
FROM ${AOB_RUNTIME_IMAGE}

COPY . /opt/suite/scenarios_data/
