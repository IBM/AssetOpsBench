"""evolve.py — the AlphaEvolve loop, adapted to the TSFM server.

AlphaEvolve (DeepMind, 2506.13131): an LLM proposes edits to a *program*, an automatic
*evaluator* scores it, and a MAP-Elites/island *database* keeps a diverse set of elites; the
loop repeats. We adopt the same machinery with one principle preserved: **the server does NOT
call an LLM** — the agent is the proposer. The server scaffolds the loop:

  evolve_ask(task)  → sample parent(s) + diverse inspirations + data evidence + the task contract
                      → the AGENT mutates them into a new candidate (recipe OR feature program).
  evolve_tell(cand) → VALIDATE (EFE gate / recipe shape) → EVALUATE (run_recipe / run_tabular /
                      feature gate → a scalar FITNESS) → place in the MAP-Elites cell / island,
                      keep the elite, record lineage (parent, generation).
  evolve_best(task) → the current elites (best per behavior cell).

The "program" (EVOLVE-BLOCK) is either a **recipe** (declarative compose spec) or an EFE
**feature program** (fit/transform/inverse code). Fitness is always higher-is-better.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ARCHIVE = "tsfm_evolve"
N_ISLANDS = 4
_FORECAST = "tsfm_forecasting"
_TABULAR = {"tsfm_regression", "tsfm_classification", "tsfm_clustering"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────── fitness + descriptor ───────────────────────
def _bucket(n: int) -> str:
    return "1" if n <= 1 else ("2-3" if n <= 3 else "4+")


def _descriptor(store, task: str, kind: str, program: dict) -> str:
    """Behaviour cell key (MAP-Elites): coarse structural signature of the candidate."""
    if kind == "feature":
        out = program.get("output_type", "?")
        inv = int(bool(program.get("invertible")))
        return f"feat|{task}|out={out}|inv={inv}"
    from . import composition as C
    regime = C._recipe_regime(program, store)
    nt = len(program.get("transforms") or [])
    ens = 1 if "ensemble" in program else 0
    nm = len(program["ensemble"]["members"]) if ens else 1
    return f"rec|{task}|reg={regime}|nc={_bucket(nt + nm)}|ens={ens}"


def _fitness_from_run(task: str, run: dict) -> float:
    """Map a run result to a scalar fitness (higher = better)."""
    if task == _FORECAST:
        return -float(run["backtest_score"])          # smape/mase: lower is better
    cv = run.get("cv_score")
    return float(cv) if isinstance(cv, (int, float)) else -1e9   # acc/r2/silhouette: higher better


# ─────────────────────────────── evaluation ─────────────────────────────────
def _evaluate(store, task: str, kind: str, program: dict, *, data_ref=None,
              timestamp_column=None, target_columns=None, label_column=None) -> dict:
    """Validate + evaluate a candidate → {valid, fitness, raw, summary}. Reuses the engine."""
    from ..io import refs
    if kind == "feature":
        return _evaluate_feature(program, data_ref=data_ref)
    from . import composition as C
    if not isinstance(program, dict) or ("estimator" not in program and "ensemble" not in program):
        return {"valid": False, "reason": "recipe must include an 'estimator' or an 'ensemble'"}
    if task == _FORECAST:
        if not data_ref or not target_columns:
            return {"valid": False, "reason": "forecasting eval needs data_ref + target_columns"}
        try:
            obj = refs.load_series(data_ref, time_col=timestamp_column, channels=target_columns)
            series = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            run = C.run_recipe(store, series, program)
        except Exception as exc:
            return {"valid": False, "reason": f"eval failed: {type(exc).__name__}: {str(exc)[:120]}"}
        return {"valid": True, "fitness": _fitness_from_run(task, run), "raw": run["backtest_score"],
                "summary": {k: run.get(k) for k in ("metric", "backtest_score", "training_regime")}}
    if task in _TABULAR:
        if not data_ref:
            return {"valid": False, "reason": "tabular eval needs data_ref"}
        try:
            df = pd.read_csv(refs._path(data_ref))
            y = None
            if label_column and label_column in df.columns:
                y = df[label_column].to_numpy(); df = df.drop(columns=[label_column])
            X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
            run = C.run_tabular_recipe(store, X, {**program, "task": task}, y=y)
        except Exception as exc:
            return {"valid": False, "reason": f"eval failed: {type(exc).__name__}: {str(exc)[:120]}"}
        cv = run.get("cv_score")
        if not isinstance(cv, (int, float)):
            return {"valid": False, "reason": f"eval failed: {cv}"}
        return {"valid": True, "fitness": float(cv), "raw": cv,
                "summary": {k: run.get(k) for k in ("task", "metric", "cv_score", "n_features")}}
    return {"valid": False, "reason": f"unsupported task '{task}'"}


def _evaluate_feature(program: dict, *, data_ref=None) -> dict:
    """Validate an EFE feature program through the gate; fitness rewards a clean, invertible op."""
    from . import feature_runner as FR
    from ..io import refs
    if data_ref:
        obj = refs.load_series(data_ref)
        X = (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_numpy(dtype=float)
    else:
        X = np.sin(np.arange(128) / 5.0).reshape(-1, 1)        # synthetic probe
    try:
        res = FR.validate_and_run(program, X, X, metadata=program.get("params") or {})
    except Exception as exc:
        return {"valid": False, "reason": f"validity gate failed: {exc}"}
    checks = res.get("checks", {})
    fitness = 1.0 + (0.5 if checks.get("invertible_ok") else 0.0)
    return {"valid": True, "fitness": fitness, "raw": fitness,
            "summary": {"checks": checks, "invertible_ok": checks.get("invertible_ok")}}


# ─────────────────────────────── archive ops ────────────────────────────────
def _elites(store, task: Optional[str] = None, kind: Optional[str] = None) -> List[dict]:
    """Best candidate per behaviour cell (the MAP-Elites grid), optionally filtered."""
    sel = {}
    if task:
        sel["task"] = task
    if kind:
        sel["kind"] = kind
    cells: Dict[str, dict] = {}
    for d in store.find(ARCHIVE, sel):
        c = d["cell"]
        if c not in cells or d["fitness"] > cells[c]["fitness"]:
            cells[c] = d
    return sorted(cells.values(), key=lambda d: d["fitness"], reverse=True)


def evolve_tell(store, task: str, kind: str, program: dict, *, parent_id: Optional[str] = None,
                data_ref=None, timestamp_column=None, target_columns=None,
                label_column=None) -> dict:
    """Validate + evaluate the agent's candidate and place it in the archive (with lineage)."""
    if kind not in ("recipe", "feature"):
        return {"accepted": False, "reason": "kind must be 'recipe' or 'feature'"}
    ev = _evaluate(store, task, kind, program, data_ref=data_ref,
                   timestamp_column=timestamp_column, target_columns=target_columns,
                   label_column=label_column)
    if not ev.get("valid"):
        return {"accepted": False, "reason": ev.get("reason", "invalid candidate")}

    cell = _descriptor(store, task, kind, program)
    parent = store.get(ARCHIVE, parent_id) if parent_id else None
    generation = int(parent["generation"]) + 1 if parent else 0
    island = parent["island"] if parent else len(store.find(ARCHIVE, {})) % N_ISLANDS
    incumbent = next((d for d in _elites(store, task, kind) if d["cell"] == cell), None)
    is_new_elite = incumbent is None or ev["fitness"] > incumbent["fitness"]

    eid = f"evo:{uuid.uuid4().hex[:10]}"
    doc = {"_id": eid, "evolve_id": eid, "task": task, "kind": kind, "program": program,
           "cell": cell, "island": island, "generation": generation, "parent_id": parent_id,
           "fitness": round(float(ev["fitness"]), 6), "raw_score": ev.get("raw"),
           "summary": ev.get("summary"), "created_at": _now()}
    store.put(ARCHIVE, doc)
    return {"accepted": True, "evolve_id": eid, "cell": cell, "island": island,
            "generation": generation, "fitness": doc["fitness"], "is_new_elite": is_new_elite,
            "incumbent_fitness": incumbent["fitness"] if incumbent else None,
            "summary": ev.get("summary")}


def evolve_ask(store, task: str, *, kind: str = "recipe", data_ref=None,
               timestamp_column=None, channels=None, n_parents: int = 2,
               n_inspirations: int = 3) -> dict:
    """Sample parents + diverse inspirations + data evidence so the agent can propose a mutation."""
    elites = _elites(store, task, kind)
    parents = elites[:n_parents]
    used = {d["cell"] for d in parents}
    inspirations, seen_islands = [], set()
    for d in elites[n_parents:]:                                   # diversity: distinct cells/islands
        if d["cell"] in used:
            continue
        if d["island"] in seen_islands and len(inspirations) >= 1:
            continue
        inspirations.append(d); used.add(d["cell"]); seen_islands.add(d["island"])
        if len(inspirations) >= n_inspirations:
            break

    evidence = None
    if data_ref:
        from ..reasoning import profile
        try:
            evidence = profile.profile_ref(data_ref, timestamp_column=timestamp_column, channels=channels)
        except Exception as exc:
            evidence = {"error": str(exc)[:120]}

    from ..core import tasks as task_spec
    contract = task_spec.get_task(task).__dict__ if task in task_spec.TASKS else None
    seed = not elites
    return {
        "task": task, "kind": kind, "archive_size": len(store.find(ARCHIVE, {"task": task})),
        "parents": [_view(d) for d in parents],
        "inspirations": [_view(d) for d in inspirations],
        "evidence": evidence, "contract": contract,
        "instructions": (
            "Propose ONE new candidate by mutating/recombining the parents (use inspirations for "
            "diversity); return it to evolve_tell. " + (
                "Archive is empty — propose an initial zero-shot recipe from discover_components."
                if seed else
                "Aim to beat the parent fitness or fill a new behaviour cell (novel structure).")),
    }


def _view(d: dict) -> dict:
    """Compact archive entry for the agent (program + where it sits + how good)."""
    return {"evolve_id": d["evolve_id"], "kind": d["kind"], "cell": d["cell"],
            "island": d["island"], "generation": d["generation"], "fitness": d["fitness"],
            "program": d["program"], "summary": d.get("summary")}


def evolve_best(store, task: str, *, kind: Optional[str] = None, top_k: int = 5) -> dict:
    """The current elites (best per behaviour cell) for a task — the evolved frontier."""
    elites = _elites(store, task, kind)[:top_k]
    return {"task": task, "kind": kind, "n_cells": len(_elites(store, task, kind)),
            "elites": [_view(d) for d in elites]}
