# HuggingGPT × sktime × file pointers — stronger stores + the recipe DAG

Re-read HuggingGPT (full paper, arXiv:2303.17580v4) and folded its real mechanisms into the
stores and the recipe, on the sktime substrate, with **IoT data passed by file pointers** (your
constraint). Built + verified (`io_refs.py`, `plan.py`, `model_store.describe_candidates`).

## 1. What HuggingGPT actually contributes (and we adopt)

| HuggingGPT mechanism (verbatim) | In our server |
|---|---|
| Task list: `[{"task","id","dep":[ids],"args":{...URL...}}]` | the **recipe DAG** (`plan.py`): steps with `id`, `dep`, `args`, `recipe` |
| `"dep"` = prior task whose **resource** this task needs | `dep` edges → topological execution |
| `"<resource>-task_id"` = the generated file from a dep task | `@step_id` in `args` → resolves to that step's **output file pointer** |
| `args` carry **URLs / file paths** (image/audio/…) | `args.data_ref` = **IoT file pointer**; never inline arrays |
| Model selection **by description**, ranked by **downloads**, top-K | `model_store.describe_candidates(task, top_k)` → ranked candidate cards |
| Hybrid endpoints (local / HF inference) | sktime resolve + **download** (`from_pretrained`); availability at call time |
| Four stages: plan → select → execute → respond | plan DAG → describe_candidates/find_models → run_plan(sktime) → result |

The two upgrades that matter most: **(a)** the recipe becomes a HuggingGPT-style DAG whose
steps chain through **file pointers** (matching how your IoT data flows), and **(b)** the model
store gains a HuggingGPT **selection surface** (descriptions + downloads), not just structured
filters.

## 2. File-pointer data model (`io_refs.py`) — IoT by reference

- IoT sensor data and every step output are **file pointers** (`file://…csv|parquet`, or s3/uri).
  Tools carry a tiny `data_ref`, never the array — exactly HuggingGPT's resource passing.
- `load_series(data_ref)` resolves a pointer → the **sktime container** (pd.Series univariate /
  pd.DataFrame multivariate), auto-detecting/dropping the time column and non-numeric columns,
  channel-subset aware.
- `write_series` / `write_json` write a step's output to a **new file pointer**; downstream
  steps reference it via `@step_id`.
- *Verified*: an IoT CSV pointer loads to a clean series; the timestamp column is dropped.

## 3. The recipe DAG (`plan.py`) — HuggingGPT task-list on sktime

```jsonc
{ "steps": [
   {"id":"f1","task":"forecast","dep":[],
    "args":{"data_ref":"file://iot_chiller6.csv","channels":["bearing_temp"]},
    "recipe":{"ensemble":{"combine":"mean","members":[{"model_id":"ttm"},{"sktime_class":"...Chronos..."}]},
              "fh":[1..12]}},
   {"id":"e1","task":"evaluate","dep":["f1"],
    "args":{"data_ref":"file://iot_chiller6.csv","by":"norm_crps","recipes":{...}}} ] }
```
`run_plan` topologically executes: resolve `@refs` to dependency file pointers → `load_series`
from the pointer → run the step on sktime (forecast = composition recipe → `EnsembleForecaster`;
anomaly = sktime detector; evaluate = GIFT-Eval leaderboard) → **write each output to a file
pointer** → persist the plan + outputs in `tsfm_plans` (state-exportable, #394).
*Verified end-to-end*: forecast step wrote `forecast_f1_*.csv`, evaluate step consumed the data
ref and wrote `eval_e1_*.json`, plan persisted.

## 4. Better stores (HuggingGPT selection + sktime download)

- **Model store**: `describe_candidates(task, top_k)` returns compact cards `{model_id,
  description, downloads, family, sktime_class, context_length, tags}` ranked by a popularity/
  quality prior (downloads, then eval metric) — the `{{Candidate Models}}` surface the agent
  reasons over (and HuggingGPT's token-saving top-K). *Verified*: TTM (50k downloads) ranks
  above the drift baseline. The chosen card's `sktime_class` is resolved+**downloaded by sktime**
  — you confirmed that's the desired path.
- **Feature store**: same selection surface for transforms (description + FLOps importance);
  the recipe's `transforms` reference them by id.

## 5. Why this is the stronger design

1. **Data scales by reference, not value.** IoT windows can be GB; the agent and tools move
   file pointers. Steps chain outputs as pointers (HuggingGPT resource dependency), so multi-step
   workflows never serialize big arrays through the LLM.
2. **The recipe is now a DAG**, not a single pipeline — multi-task workflows (forecast → detect
   on residuals → evaluate; or per-asset fan-out) are first-class, with dependencies + resource
   passing.
3. **Selection is description-driven** (HuggingGPT) AND structured-tag-driven (sktime) AND
   budget-driven (T-Daub) — three complementary rankers over one catalog.
4. **sktime downloads the models** — no bespoke loaders; a card's `sktime_class` is fetched on
   demand; availability resolves at call time.
5. **GIFT-Eval scores it** — an `evaluate` step in the DAG runs the leaderboard.

## 6. Verified
`tests/test_plan.py` (4): IoT-as-file-pointer load, `@resource` resolution, forecast→evaluate
DAG with file-pointer outputs + persisted plan, HuggingGPT candidate ranking. Plus the prior
suites (composition, gifteval, stores) green.

## Files
`io_refs.py` (file-pointer I/O), `plan.py` (recipe DAG), `model_store.describe_candidates`
(HuggingGPT selection), composing with `composition.py` (recipes/ensembles), `gifteval.py`
(scoring), `sktime_resolver.py` (substrate + download).

## Source
HuggingGPT (arXiv:2303.17580v4) — task list `{task,id,dep,args}`, `<resource>-task_id`,
model selection by description ranked by downloads, hybrid endpoints, 4-stage workflow.
