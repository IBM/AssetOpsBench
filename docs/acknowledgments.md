# Acknowledgments

AssetOpsBench builds on ideas from several prior benchmark and evaluation efforts. We acknowledge these works for shaping the design of trajectory-based evaluation, multi-metric reporting, deterministic scoring, and agent benchmark workflows.

## Evaluation and Benchmark Design

AssetOpsBench's offline evaluation workflow is inspired by benchmark designs that separate task execution from scoring. In particular, the evaluation pipeline follows the pattern:

```text
agent run → saved trajectory/output → evaluator → report
```

This design allows saved agent trajectories to be re-scored with different scorers or judge models without re-running the agent.

We acknowledge the following benchmark efforts:

* **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**
  Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan.
  SWE-bench motivates task-level evaluation from saved model outputs in realistic software engineering environments.
  https://arxiv.org/abs/2310.06770

* **HELM: Holistic Evaluation of Language Models**
  Percy Liang et al.
  HELM motivates standardized, multi-metric evaluation and transparent reporting across scenarios, models, and metrics.
  https://arxiv.org/abs/2211.09110

* **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**
  Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan.
  τ-bench motivates evaluating tool-using agents through realistic interaction traces and final-state/task-success scoring.
  https://arxiv.org/abs/2406.12045

## Static Structured-Answer Scoring

The `static_json` scorer is inspired by the static scoring implementation in the DeepSynth benchmark, particularly the structured-answer parsing, normalization, flattening, exact/partial matching, and precision/recall/F1-style evaluation ideas from:

* **DeepSynth Benchmark — `eval_static_score.py`**
  https://github.com/agentdeepsynthesis/deepsynth-bench/blob/main/scripts/evaluation/eval_static_score.py

The AssetOpsBench implementation adapts these ideas to the existing trajectory-based evaluation pipeline. Instead of evaluating a standalone model-output file, AssetOpsBench joins saved agent trajectories with scenario ground truth and invokes the scorer through:

```bash
uv run evaluate \
  --trajectories traces/trajectories \
  --scenarios /path/to/scenarios_data \
  --scorer-default static_json
```

## Implementation Note

The references above are acknowledged for conceptual and methodological inspiration. AssetOpsBench implements its own evaluation interfaces, scorer registry, trajectory loading, scenario loading, and report generation to support industrial asset operations and maintenance benchmarks.
