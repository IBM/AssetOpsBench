# Task template

`benchmarks/harbor/adapter/generate_tasks.py` copies these files into every generated
task, substituting the scenario id, category and scorer.

Nothing scenario-specific lives here. The adapter adds, per scenario:

- `instruction.md`               from the scenario's `question.txt`
- `environment/scenario_<id>/`   the per-task image layer's build context
- `tests/scenarios/scenario_<id>/` ground truth and scorer inputs
- `solution/{question,groundtruth}.txt` for the oracle
