# TSFM MCP Server — documentation index & current design (canonical)

The design evolved through several critique cycles. This index is the **single source of
truth** for what's current vs superseded, plus the canonical 12-line architecture. Read it
first.

## Current architecture (12 lines)

1. **Substrate = sktime.** Every model/transform is an sktime estimator (or sklearn/PyOD via
   adapter, or custom `Base*`); sktime downloads weights. (`sktime_resolver.py`)
2. **Core = a CouchDB/Memory store** (`store.py`) holding cards + result/run tables;
   `export_state()` snapshots everything for scoring (#394).
3. **Model store** = sktime-card catalog (superset of the registry): find/select/register/
   lineage. (`model_store.py`, `schemas.py`) — see **STORES.md**.
4. **Feature store** = transforms (sktime/EFE) + FLOps extractors: provide/generate/select.
   (`feature_store.py`, `feature_selection.py`, `feature_runner.py`) — see **STORES.md**.
5. **8 standardized TS-AI tasks** bound to sktime scitypes. (`task_spec.py`)
6. **Data by file pointer.** Agent downloads a window → CSV → `data_ref`; never inline.
   (`io_refs.py`)
7. **Recipe** = an sktime-compiled spec (transforms + single/ensemble + conformal + eval).
   (`composition.py`)
8. **Recipe DAG** = HuggingGPT task-list: steps with `dep` + `@resource` file-pointer chaining.
   (`plan.py`)
9. **Selection (plural):** description/HuggingGPT (`describe_candidates`), tags (sktime),
   budget T-Daub (`selector.py`).
10. **Scoring = GIFT-Eval**: seasonal-naive-normalized MASE+CRPS, geo-mean, mean-rank
    leaderboard. (`gifteval.py`)
11. **Reasoning is the agent's**: server gives evidence (`profile.py`), agent decides, server
    grades (`scoring.py`). (`REASONING_OWNERSHIP.md`)
12. **Agent loop:** discover candidates → compose recipe/DAG (file pointers) → run on sktime →
    GIFT-Eval score → iterate (lineage persisted).

## Document status

| Doc | Status | What it is |
|---|---|---|
| **STORES.md** | ✅ authoritative | the definitive Model + Feature store spec (read this) |
| **DOCS_INDEX.md** | ✅ authoritative | this file — current design + status map |
| **SKTIME_NATIVE_DESIGN.md** | ✅ authoritative | the sktime-substrate pivot (the decision) |
| **CORE_CAPABILITY.md** | ✅ authoritative | the agentic compose→run→diagnose→iterate loop |
| **GIFTEVAL_APPROACH.md** | ✅ authoritative | evaluation backbone (MASE/CRPS, seasonal-naive norm, rank) |
| **HUGGINGGPT_FILEPOINTER_DESIGN.md** | ✅ authoritative | recipe DAG + file pointers + candidate selection |
| **CAPABILITY_MATRIX.md** | ✅ authoritative | conformal / sklearn-PyOD / TSPulse coverage check (verified) |
| **REASONING_OWNERSHIP.md** | ✅ authoritative | agent reasons; server = evidence + grader |
| **README.md** | ✅ current | quick start + module map (sktime-native) |
| **DESIGN.md** | 🟡 historical | the unified 8-task abstraction + T-Daub; correct, but the
  estimator/transform contract is now sktime (see SKTIME_NATIVE_DESIGN). Task taxonomy + T-Daub
  + GIFT-Eval framing still hold. |
| **BUILDING_BLOCK.md** | ⛔ superseded | the custom typed Operator algebra. **sktime base
  classes replace it.** Kept for the *ideas* (leakage-safe fit/transform, invertibility,
  quality-gate, zero-model) — all now realized via sktime. |
| **TOOLS_ARE_COMPLEX.md** | 🟡 historical | the parameter-reasoning argument; superseded in
  *ownership* by REASONING_OWNERSHIP (agent decides, server grades), but the decision-space
  tables are still useful. |

## Superseded / dropped (do not build)
- **`operators.py`** (custom algebra) → sktime base classes. **`planner.py`** server-side
  parameter planner → demoted to optional advisor (agent reasons; `scoring.py` grades).
- **External AI4Industry AD service** → in-toolkit (AnomalyKiTS detectors + TSPulse + PyOD),
  all in sktime.
- **Inline data** → file pointers (`io_refs.py`).

## Module → doc map
- stores: `model_store.py`,`feature_store.py`,`schemas.py` → STORES.md
- substrate: `sktime_resolver.py`,`store.py` → SKTIME_NATIVE_DESIGN.md
- recipe/loop: `composition.py`,`plan.py`,`io_refs.py` → CORE_CAPABILITY / HUGGINGGPT_FILEPOINTER
- features: `feature_selection.py`(FLOps),`feature_runner.py`(EFE) → STORES.md
- selection: `selector.py`(T-Daub) → DESIGN.md §5 / STORES §1.3
- eval: `gifteval.py` → GIFTEVAL_APPROACH.md
- reasoning: `profile.py`,`scoring.py` → REASONING_OWNERSHIP.md
- tasks: `task_spec.py` → DESIGN.md §2 / STORES
- (legacy: `operators.py`,`runner.py`,`compute.py`,`pipeline.py`,`io.py` — pre-sktime; keep
  `runner.py` availability logic, fold the rest into the sktime path)

## Provenance — the four papers, mapped
- **AutoAI-TS**: zero-model, invertible transform stacks, **T-Daub** selection → `selector.py`,
  sktime `TransformedTargetForecaster` / `NaiveForecaster`.
- **AnomalyKiTS**: AD pipelines + Static/Dynamic thresholding + EM/AL → sktime `detector`s.
- **FLOps**: typed extractor library + reference-feature/CD selection → `feature_selection.py`.
- **EFE**: evolved fit/transform programs, validity-gated, archived → `feature_runner.py` +
  feature store lineage.
- **HuggingGPT**: task-DAG with resource (file-pointer) dependencies + model selection by
  description → `plan.py` + `model_store.describe_candidates`.
- **GIFT-Eval**: seasonal-naive-normalized MASE/CRPS + rank aggregation → `gifteval.py`.

## Test status
38+ tests green across: stores, schemas, FLOps select, EFE validity gate, task_spec, T-Daub,
composition loop (ensemble + iterate), GIFT-Eval leaderboard, conformal/PyOD/TSPulse resolution,
file-pointer DAG, candidate ranking.
