# Glossary — how the agent learns the vocabulary

The server uses a small, consistent vocabulary. To make sure an agent never has to guess what a
term means, the definitions live in **one place** (`core/glossary.py`) and are surfaced through
**three channels** the agent always sees:

1. **Server instructions** (loaded with the toolset) — a one-line-per-term `VOCABULARY` summary
   + the `WORKFLOW`, via `glossary.short_glossary()`.
2. **`discover_components(task)`** — returns the full `glossary`, `workflow`, and `principles`
   inline (the natural "menu" tool the agent calls first).
3. **Tool docstrings** — each of tools 1–7 defines the term it deals in (in CAPS) and points to
   the next step; unknown-`task_id` errors list the valid tasks.

## Terms

| Term | Meaning | Where |
|------|---------|-------|
| **task** | one of 8 standardized TS-AI problem types (forecasting, regression, classification, anomaly_detection, imputation, evaluation, similarity_search, clustering) | `list_tasks` |
| **component** | any catalog entry you place in a recipe — a model OR a feature (DATA, not a tool) | `get_component` |
| **model** | an estimator card pointing at an sktime / foundation class | `find_models` |
| **candidate** | a model proposed + ranked for a task (a shortlist; you still choose) | `describe_candidates` |
| **feature** | a transform/extractor card (normalization, lag, catch22, FLOps set) | `find_features` |
| **transform** | a feature used as a preprocessing/extraction step in a recipe (some invertible) | `find_features` |
| **ensemble** | a recipe combining several models (mean/median/weighted/stack) | `run_recipe` |
| **recipe** | the declarative spec you author: transforms + model/ensemble + conformal/finetune/anomaly blocks + eval | `run_recipe` / `run_tabular_recipe` / `run_plan` |
| **training_regime** | zero_shot (default) / fit_on_series / fine_tune | `discover_components` |
| **param_schema** | per-model parameter hints + ranges you reason from the data | `get_component` |
| **data_ref** | data passed BY REFERENCE — a file pointer (`dataset_path`); results return as a `results_file` pointer | `profile_series` / `run_recipe` |
| **evidence** | facts about the data you reason from (server states facts, you decide) | `profile_series` |
| **result** / **run** | a written per-task output record / a recipe-execution record with provenance + audit | `get_result` / `get_run` |

## Workflow
1. `list_tasks` — pick the task.
2. `profile_series(dataset_path)` — read the evidence.
3. `discover_components(task)` / `describe_candidates` / `find_models` / `find_features` — see the menu.
4. `get_component(id)` — read a card + its `param_schema`; reason the parameters.
5. author a recipe → `run_recipe` / `run_tabular_recipe` / `run_plan` (zero-shot first).
6. `evaluate` (GIFT-Eval) → inspect `get_run` → revise and iterate.

## Principles
Models & features are DATA composed via recipes · data and results cross as file pointers ·
the agent reasons every choice, the server gives evidence + grades · zero-shot is the default.

_Source of truth: `core/glossary.py`. Tested in `tests/test_glossary.py`._
