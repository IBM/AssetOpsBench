from __future__ import annotations

from typing import Dict, List

# term → (definition, how it relates to the others / which tool exposes it)
TERMS: Dict[str, Dict[str, str]] = {
    "task": {
        "definition": "a standardized TS-AI problem type (8 total: forecasting, regression, "
        "classification, anomaly_detection, imputation, evaluation, "
        "similarity_search, clustering). Defines required inputs, output, eval.",
        "tool": "list_tasks",
    },
    "component": {
        "definition": "any catalog entry you can place in a recipe — a MODEL or a FEATURE. "
        "Components are DATA (cards), not tools.",
        "tool": "get_component",
    },
    "model": {
        "definition": "an estimator card (forecaster / classifier / detector / clusterer) that "
        "points at an sktime or foundation-model class.",
        "tool": "find_models",
    },
    "candidate": {
        "definition": "a model proposed AND ranked for a task (HuggingGPT-style, by description + "
        "popularity). A shortlist — you still choose.",
        "tool": "describe_candidates",
    },
    "feature": {
        "definition": "a transform/extractor card (normalization, lag/rolling, catch22, a "
        "FLOps-selected set). Applied before the estimator.",
        "tool": "find_features",
    },
    "transform": {
        "definition": "a feature used as a preprocessing/extraction step inside a recipe; some "
        "are invertible (round-trip back to input space).",
        "tool": "find_features",
    },
    "ensemble": {
        "definition": "a recipe that combines several models (mean / median / weighted / stack).",
        "tool": "run_recipe",
    },
    "recipe": {
        "definition": "the declarative spec YOU author: transforms + a single model OR an "
        "ensemble + optional conformal/finetune/anomaly blocks + eval protocol.",
        "tool": "run_recipe / run_tabular_recipe / run_plan",
    },
    "training_regime": {
        "definition": "how much training a model needs: zero_shot (pretrained, no training — the "
        "default) | fit_on_series | fine_tune.",
        "tool": "discover_components",
    },
    "param_schema": {
        "definition": "per-model parameter hints + ranges you reason over (context_length, sp, "
        "n_neighbors, …) from the data evidence.",
        "tool": "get_component",
    },
    "data_ref": {
        "definition": "data is passed BY REFERENCE — a file pointer (dataset_path: a CSV/parquet "
        "path or file:// URI), never inlined. Results return as a results_file pointer.",
        "tool": "profile_series / run_recipe",
    },
    "evidence": {
        "definition": "the facts about the data you reason from (seasonality, stationarity, "
        "channels, length). The server states facts; YOU decide.",
        "tool": "profile_series",
    },
    "result": {
        "definition": "a written per-task output record in a result table (forecast/anomaly/…).",
        "tool": "get_result / list_results",
    },
    "run": {
        "definition": "a recipe-execution record with provenance, param_audit and lineage.",
        "tool": "get_run / list_runs",
    },
    "evolve": {
        "definition": "an AlphaEvolve-style loop: ask for parents+inspirations from an archive of "
        "elites (best program per behaviour cell), mutate into a new candidate, tell "
        "it back to be validated/scored/archived. You propose; the server grades.",
        "tool": "evolve_ask / evolve_tell / evolve_best",
    },
}

# the canonical sequence — what to call, in order
WORKFLOW: List[str] = [
    "1. list_tasks — pick the task (defines inputs/output/eval).",
    "2. profile_series(dataset_path) — read the data evidence.",
    "3. discover_components(task) / describe_candidates / find_models / find_features — see the menu.",
    "4. get_component(id) — read a card + its param_schema; reason the parameters.",
    "5. author a recipe → run_recipe / run_tabular_recipe / run_plan (zero-shot first).",
    "6. evaluate (GIFT-Eval) → inspect get_run → revise the recipe and iterate.",
]

PRINCIPLES: List[str] = [
    "Models & features are DATA (catalog cards), not tools — composed via recipes.",
    "Data crosses the boundary as file pointers; results come back as file pointers.",
    "The agent reasons every choice; the server gives evidence + grades, it does not decide.",
    "Zero-shot is the default; fine-tune is an explicit, optional escalation.",
]


def glossary() -> dict:
    """The full vocabulary + workflow + principles — safe to embed anywhere the agent reads."""
    return {"terms": TERMS, "workflow": WORKFLOW, "principles": PRINCIPLES}


def short_glossary() -> str:
    """A compact one-line-per-term string for the server instructions (always in context)."""
    return " · ".join(f"{k}: {v['definition'].split('.')[0]}" for k, v in TERMS.items())
