# AlphaEvolve alignment — and how we support Evolve

AlphaEvolve (DeepMind, arXiv 2506.13131) is an evolutionary coding agent: an LLM proposes edits
to a *program*, an automatic *evaluator* scores it, and a MAP-Elites / island *database* keeps a
diverse set of elites; the loop repeats. Our server adopts the same machinery, with one
principle preserved — **the server never calls an LLM; the agent is the proposer.**

## Component mapping

| AlphaEvolve | TSFM server | Where |
|---|---|---|
| **EVOLVE-BLOCK** (the program) | a **recipe** (declarative compose spec) OR an EFE **feature program** (fit/transform/inverse) | `evolve_tell(kind=...)` |
| **`evaluate(h)`** evaluator | `run_recipe` / `run_tabular_recipe` (→ backtest/CV) and the EFE validity gate → a scalar **fitness** | `engine/evolve._evaluate` |
| **"avoid incorrect suggestions"** | EFE gate (entry points, no-inplace, invertibility round-trip) + recipe-shape check; invalid ⇒ rejected, not archived | `feature_runner`, `_evaluate` |
| **Program database** (MAP-Elites + islands) | archive keyed by **behaviour cell** (task × regime × #components × ensemble; or out_type × invertible) + **islands** for diversity; best-per-cell = elite | `engine/evolve` (`ARCHIVE`) |
| **Prompt sampling** (parents + inspirations + context) | `evolve_ask` samples top parents + diverse inspirations (distinct cells/islands) + `profile_series` evidence + task contract | `evolve_ask` |
| **LLM ensemble** (proposer) | the **agent** — outside the server, by design | (agent) |
| **Evolutionary lineage** | `parent_id` + `generation` per candidate | archive docs |
| **Evaluation cascade** | reuses T-Daub reverse-allocation; cheap configs first (future) | `selector` |

## The loop (agent-generates / server-grades)
```
evolve_ask(task[, dataset_path])   → {parents, inspirations, evidence, contract, instructions}
        ↓  the AGENT mutates/recombines the parents into ONE new candidate (recipe or feature)
evolve_tell(task, kind, program[, parent_id, dataset_path, …])
        → validate (gate/shape) → evaluate → fitness → MAP-Elites cell/island + lineage
        → {accepted, fitness, is_new_elite, cell, generation}
evolve_best(task)                  → the elites (best program per behaviour cell)
```
Fitness is always higher-is-better (forecasting: −backtest error; tabular: CV accuracy/r²/
silhouette; feature: validity + invertibility bonus). Different *structure* lands in a different
cell, so the archive keeps a diverse frontier rather than collapsing to one design.

## What we evolve
- **Recipes** — which models/transforms/ensemble/params to compose (compose-level search).
- **Feature programs** — EFE `fit/transform/inverse` code the agent writes; gated + archived.

## Status
Built + tested (`engine/evolve.py`, `tests/test_evolve.py`, 6 tests): seed → grow archive across
cells, elite replacement within a cell, ask returns parents/inspirations/evidence, empty-archive
seed instructions, invalid-candidate rejection, feature-program evolution through the gate. MCP
tools `evolve_ask` / `evolve_tell` / `evolve_best` are live (27 tools total).

## Gaps vs full AlphaEvolve (future)
- **Evaluation cascade** as an explicit evolve gate (staged cheap→expensive configs).
- **Island migration** (periodic cross-island copying) — islands are stored + used for sampling
  diversity, but migration is not yet scheduled.
- **Multi-objective elites** (Pareto front over accuracy × cost/latency) — today fitness is one
  scalar per cell.

Sources: AlphaEvolve — arXiv 2506.13131 (Novikov et al., DeepMind, 2025).
