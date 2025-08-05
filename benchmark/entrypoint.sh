#!/bin/bash
# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate assetopsbench

which python
python --version
python -m pip show agent_hive
python -m pip show reactxen
python -m pip show fmsr_agent
python -m pip show iotagent
python -m pip show tsfmagent

# Run the entire thing
python /home/run_plan_execute.py --utterance_file /home/scenarios/multi_agent/end2end_utterance.json --llm 6 --workflow p

# Keep the container alive
tail -f /dev/null
