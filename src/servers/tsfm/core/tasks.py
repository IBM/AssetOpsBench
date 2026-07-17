from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TSTask:
    task_id: str
    output_verb: str  # predict | score | transform | assign | metrics
    output_type: str
    eval_protocol: str  # backtest | blocked_cv | stratified_blocked_cv |
    # mask_and_score | range_pr | retrieval_at_k | silhouette | meta
    metrics: List[str]
    supervised: bool
    selection_signal: (
        str  # mase | f1 | auc_pr | em_al | silhouette | recall_at_k | meta
    )
    result_collection: str
    required_inputs: List[str]  # keys a request must supply
    requires_inverse: bool = (
        False  # transforms must round-trip (output back in input space)
    )
    leakage_split: str = "blocked"  # blocked | stratified_blocked | none(meta)
    notes: str = ""
    description: str = ""  # plain-language: what this task does (for the agent)


TASKS: Dict[str, TSTask] = {
    "tsfm_forecasting": TSTask(
        "tsfm_forecasting",
        "predict",
        "forecast",
        "backtest",
        ["mase", "wql", "smape", "mae"],
        True,
        "mase",
        "forecast_result",
        ["series", "horizon"],
        requires_inverse=True,
        notes="AutoAI-TS pipelines + TSFMs; horizon from the request; invertible scaling/flatten",
        description="Predict future values of a series over a horizon.",
    ),
    "tsfm_regression": TSTask(
        "tsfm_regression",
        "predict",
        "value",
        "blocked_cv",
        ["r2", "mae", "rmse"],
        True,
        "r2",
        "regression_result",
        ["series", "target"],
        notes="FLOps feature extraction+selection → regressor (RUL is a variant)",
        description="Predict a real-valued target from a series (e.g. remaining useful life).",
    ),
    "tsfm_classification": TSTask(
        "tsfm_classification",
        "predict",
        "class",
        "stratified_blocked_cv",
        ["accuracy", "f1", "auroc"],
        True,
        "f1",
        "classification_result",
        ["series", "labels"],
        leakage_split="stratified_blocked",
        notes="catch22/representation features → classifier (PHM fault classification)",
        description="Assign a discrete class/label to a series (e.g. fault type).",
    ),
    "tsfm_anomaly_detection": TSTask(
        "tsfm_anomaly_detection",
        "score",
        "score_label_contribution",
        "range_pr",
        ["auc_pr", "range_f1"],
        False,
        "em_al",
        "anomaly_result",
        ["series"],
        notes="AnomalyKiTS pipelines; label-free EM/AL when no GT; static/dynamic threshold",
        description="Score/flag abnormal points or ranges in a series.",
    ),
    "tsfm_imputation": TSTask(
        "tsfm_imputation",
        "predict",
        "filled_series",
        "mask_and_score",
        ["mae", "crps"],
        True,
        "mae",
        "imputation_result",
        ["series", "mask"],
        requires_inverse=True,
        notes="TSPulse/MOMENT; hide observed points and score reconstruction",
        description="Fill missing values in a series.",
    ),
    "tsfm_evaluation": TSTask(
        "tsfm_evaluation",
        "metrics",
        "metrics",
        "meta",
        ["mae", "rmse"],
        True,
        "meta",
        "evaluation_result",
        ["predictions", "ground_truth"],
        leakage_split="none",
        notes="meta task: score predictions / benchmark pipelines",
        description="Score predictions against ground truth / benchmark pipelines.",
    ),
    "tsfm_similarity_search": TSTask(
        "tsfm_similarity_search",
        "transform",
        "embedding",
        "retrieval_at_k",
        ["recall_at_k", "map"],
        False,
        "recall_at_k",
        "similarity_result",
        ["series", "query"],
        notes="TSPulse embeddings → vector index → top-k",
        description="Find the series most similar to a query (embedding + top-k retrieval).",
    ),
    "tsfm_clustering": TSTask(
        "tsfm_clustering",
        "assign",
        "assignments",
        "silhouette",
        ["silhouette", "ari"],
        False,
        "silhouette",
        "clustering_result",
        ["series_set"],
        notes="catch22/embedding features → clusterer; label-free silhouette",
        description="Group similar series into clusters (unsupervised).",
    ),
}


def get_task(task_id: str) -> TSTask:
    if task_id not in TASKS:
        raise ValueError(f"unknown task '{task_id}'. Known: {list(TASKS)}")
    return TASKS[task_id]


def list_tasks() -> List[dict]:
    return [t.__dict__ for t in TASKS.values()]
