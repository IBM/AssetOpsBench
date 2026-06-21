"""selector.py — budget-aware pipeline selection.

T-Daub (AutoAI-TS, SIGMOD'21): rank many candidate pipelines on sequential data without
training every one to completion. Reverse *progressive data allocation*:

  1. Fixed allocation — train each pipeline on increasingly long *most-recent* slices of the
     train set, score on a fixed holdout; fit a linear learning curve over (allocation, score)
     and PROJECT the score at full length L; rank by the projection.
  2. Acceleration — give geometrically increasing data only to the top pipeline(s); re-project
     and re-rank.
  3. Output ranked pipelines (train the winner to completion downstream).

For unsupervised tasks (no labels) selection switches to label-free ranking (AnomalyKiTS
EM/AL, or silhouette) — same interface, signal-appropriate.

`score_fn(pipeline_id, n_recent) -> error` (lower is better) abstracts "train on the most
recent n points, score on holdout"; the caller supplies it (Stub for tests, real training in
production). The point verified here: T-Daub finds the best pipeline using far less total data
than training all to completion.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _project_to_full(allocs: List[int], scores: List[float], L: int) -> float:
    """Linear learning-curve fit over (allocation, score); predict score at full length L."""
    if len(allocs) == 1:
        return scores[0]
    a = np.asarray(allocs, float); s = np.asarray(scores, float)
    m, b = np.polyfit(a, s, 1)
    return float(m * L + b)


def tdaub_select(pipelines: List[str], L: int, score_fn: Callable[[str, int], float], *,
                 min_allocation: Optional[int] = None, fixed_runs: int = 4,
                 geo_increment: float = 2.0, top_k: int = 3,
                 run_to_completion: int = 1) -> dict:
    """Return ranked pipelines + the projected scores + total data 'spent' (for budget proof).

    The budget win scales with the number of candidate pipelines: every pipeline pays only the
    cheap fixed-allocation part; only the top_k get full-length data. top_k must be large
    enough that the true best survives the projection — a known T-Daub robustness knob.
    """
    min_allocation = min_allocation or max(L // 40, 8)
    spent = 0
    curves: Dict[str, Tuple[List[int], List[float]]] = {p: ([], []) for p in pipelines}

    # 1) fixed allocation — recent slices of geometrically/linearly increasing size
    for i in range(1, fixed_runs + 1):
        n = min(min_allocation * i, L)
        for p in pipelines:
            err = score_fn(p, n); spent += n
            curves[p][0].append(n); curves[p][1].append(err)
    proj = {p: _project_to_full(*curves[p], L) for p in pipelines}     # reported, auxiliary
    # Rank the fixed phase by the score at the LARGEST observed allocation, not by the linear
    # extrapolation: extrapolating an exponential learning curve overshoots and would favor
    # steeper (worse) curves. Most-data-observed is the robust ranking signal.
    eff = {p: curves[p][1][-1] for p in pipelines}
    ranked = sorted(pipelines, key=lambda p: eff[p])            # lower error = better

    # 2) acceleration — only the top pipelines get more (recent) data, geometrically, to full L
    for p in ranked[:max(top_k, 1)]:
        n = curves[p][0][-1]
        while n < L:
            n = min(int(n * geo_increment), L)
            err = score_fn(p, n); spent += n
            curves[p][0].append(n); curves[p][1].append(err)
        eff[p] = curves[p][1][-1]                               # truthful score measured at L
    ranked = sorted(pipelines, key=lambda p: eff[p])

    # 3) (optionally) the winner is trained to completion downstream
    full_budget = len(pipelines) * L                            # naive: train all on all data
    return {"ranked": ranked, "effective_score": {p: round(eff[p], 4) for p in ranked},
            "projected_score": {p: round(proj[p], 4) for p in ranked},
            "winner": ranked[0], "data_spent": spent, "naive_full_budget": full_budget,
            "budget_fraction": round(spent / full_budget, 3)}


def label_free_rank(pipelines: List[str], signal_fn: Callable[[str], float],
                    higher_is_better: bool = True) -> dict:
    """Unsupervised selection: rank by EM/AL or silhouette (AnomalyKiTS). No labels needed."""
    scores = {p: signal_fn(p) for p in pipelines}
    ranked = sorted(pipelines, key=lambda p: scores[p], reverse=higher_is_better)
    return {"ranked": ranked, "scores": {p: round(scores[p], 4) for p in ranked},
            "winner": ranked[0]}


def select(task_supervised: bool, pipelines: List[str], **kw) -> dict:
    """Uniform entry: T-Daub for supervised, label-free ranking otherwise."""
    if task_supervised:
        return tdaub_select(pipelines, kw["L"], kw["score_fn"],
                            min_allocation=kw.get("min_allocation"),
                            fixed_runs=kw.get("fixed_runs", 5), top_k=kw.get("top_k", 1))
    return label_free_rank(pipelines, kw["signal_fn"], kw.get("higher_is_better", True))
