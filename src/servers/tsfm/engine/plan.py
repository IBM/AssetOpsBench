"""plan.py — the recipe DAG (HuggingGPT task-list, on sktime, with file-pointer resources).

A PLAN is HuggingGPT's task list:
  [{"id": "s1", "task": "forecast", "dep": [], "args": {"data_ref": "file://iot.csv", ...},
    "recipe": {...composition recipe...}},
   {"id": "s2", "task": "evaluate", "dep": ["s1"], "args": {"pred": "@s1", "truth_ref": "..."}}]

- `args` carry **file pointers** (IoT data, `data_ref`) or **resource references** `@step_id`
  that resolve to the output file pointer of a dependency step (HuggingGPT's `<resource>-id`).
- The server topologically executes the DAG: load inputs from file pointers, run the step on
  the sktime substrate, write each output to a NEW file pointer, and pass refs downstream.
- Every step output and the run record are file pointers / CouchDB rows → state-exportable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from tsfm.io import refs as io_refs
from tsfm.engine import composition as C
from tsfm.eval import gifteval as G


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _toposort(steps: List[dict]) -> List[dict]:
    by_id = {s["id"]: s for s in steps}
    seen, order = set(), []
    def visit(s):
        if s["id"] in seen:
            return
        for d in s.get("dep", []):
            visit(by_id[d])
        seen.add(s["id"]); order.append(s)
    for s in steps:
        visit(s)
    return order


def _resolve_arg(v, outputs: Dict[str, dict]):
    """@step_id → that step's output file pointer; otherwise pass through (a file pointer/literal)."""
    if isinstance(v, str) and v.startswith("@"):
        return outputs[v[1:]]["ref"]
    return v


def run_plan(store, plan: dict, *, asset_id: str = "asset", scenario_id: str = None) -> dict:
    """Execute a recipe DAG. Returns per-step outputs (file pointers + summaries) + a run id."""
    steps = _toposort(plan["steps"])
    outputs: Dict[str, dict] = {}
    plan_id = f"plan:{uuid.uuid4().hex[:10]}"

    for s in steps:
        task = s["task"]
        args = {k: _resolve_arg(v, outputs) for k, v in (s.get("args") or {}).items()}

        if task == "forecast":
            y = io_refs.load_series(args["data_ref"], value_col=args.get("value_col"),
                                    time_col=args.get("time_col"), channels=args.get("channels"))
            fc = C.build_forecaster(s["recipe"], store)
            fh = s["recipe"].get("fh", list(range(1, 13)))
            fc.fit(pd.Series(np.asarray(y, float).ravel()), fh=fh)
            pred = fc.predict()
            ref = io_refs.write_series(pred, name=f"forecast_{s['id']}")
            outputs[s["id"]] = {"ref": ref, "task": task,
                                "summary": {"horizon": len(fh), "head": pred.round(4).head(3).values.tolist()}}

        elif task == "anomaly":
            X = io_refs.load_series(args["data_ref"], channels=args.get("channels"))
            from tsfm.substrate import resolver as R
            det = R.resolve(s["recipe"]["estimator"])
            Xf = pd.DataFrame(np.asarray(X, float).reshape(len(X), -1))
            det.fit(Xf)
            try:
                labels = det.predict(Xf)
                n_anom = int(np.asarray(labels).astype(bool).sum())
            except Exception as e:
                labels, n_anom = None, f"err:{type(e).__name__}"
            ref = io_refs.write_json({"labels": np.asarray(labels).tolist() if labels is not None else None}, name=f"anomaly_{s['id']}")
            outputs[s["id"]] = {"ref": ref, "task": task, "summary": {"anomaly_count": n_anom}}

        elif task == "evaluate":
            cfgs = args["configs"] if "configs" in args else [{
                "name": asset_id, "y": np.asarray(io_refs.load_series(args["data_ref"]), float).ravel().tolist(),
                "fh": args.get("fh", list(range(1, 13))), "sp": args.get("sp", 1)}]
            recipes = args.get("recipes") or {s["id"]: s.get("recipe", {})}
            board = G.leaderboard(store, recipes, cfgs, by=args.get("by", "norm_mase"))
            ref = io_refs.write_json(board, name=f"eval_{s['id']}")
            outputs[s["id"]] = {"ref": ref, "task": task, "summary": board["leaderboard"][:3]}

        else:
            outputs[s["id"]] = {"ref": None, "task": task, "summary": {"error": f"unknown task {task}"}}

    rec = {"_id": plan_id, "plan_id": plan_id, "asset_id": asset_id, "scenario_id": scenario_id,
           "plan": plan, "outputs": {k: {"ref": v["ref"], "summary": v["summary"]} for k, v in outputs.items()},
           "created_at": _now()}
    if store is not None:
        store.put("tsfm_plans", rec)
    return {"plan_id": plan_id, "steps": list(outputs), "outputs": rec["outputs"]}
