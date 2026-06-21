"""param_space.py — per-model parameter schema + reasoning + validation.

Every model has its own parameters whose VALUES must be reasoned (context_length, sp,
n_neighbors, n_clusters, strategy, …). A card therefore exposes a parameter schema:
  - auto-introspected from the sktime class (names, defaults, types, example sets), plus
  - curated `param_hints` per parameter: a one-line description, what data evidence it
    `depends_on`, a `suggest` rule, and an allowed `range`/`choices`.

The agent reads the schema + `profile_series` evidence, REASONS a value for each parameter,
fills the recipe's `params`, and the server VALIDATES them (and the scorer grades them). This
is the "tools are complex" principle applied to the full per-model parameter space.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional


def introspect(sktime_class: str) -> dict:
    """Constructor params (name → default/required/type) + sktime example param sets."""
    module, cls = sktime_class.rsplit(".", 1)
    Est = getattr(importlib.import_module(module), cls)
    sig = inspect.signature(Est.__init__)
    params: Dict[str, dict] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        params[name] = {
            "default": (None if p.default is inspect._empty else _jsonable(p.default)),
            "required": p.default is inspect._empty,
            "type": (None if p.annotation is inspect._empty else _typename(p.annotation)),
        }
    examples = None
    try:
        examples = Est.get_test_params()                     # sktime: valid example configs
    except Exception:
        pass
    return {"sktime_class": sktime_class, "params": params, "examples": examples}


def _typename(a):
    return getattr(a, "__name__", str(a))


def _jsonable(v):
    return v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)


# curated reasoning hints for common TS params — what evidence drives the value
DEFAULT_HINTS = {
    "context_length": {"description": "input/look-back window the model sees",
                       "depends_on": "dominant_period (seasonality)",
                       "suggest": ">= 2x dominant_period; must cover the seasonal cycle",
                       "range": [8, 2048]},
    "prediction_length": {"description": "forecast horizon", "depends_on": "the request",
                          "suggest": "match the horizon named in the task", "range": [1, 1024]},
    "sp": {"description": "seasonal periodicity", "depends_on": "dominant_period",
           "suggest": "= dominant_period from profile_series", "range": [1, 1024]},
    "window_length": {"description": "rolling/reduction window", "depends_on": "dominant_period",
                      "suggest": "~1-2x dominant_period", "range": [2, 1024]},
    "strategy": {"description": "naive strategy", "choices": ["last", "mean", "drift"],
                 "suggest": "drift if trending, last if persistent, mean if stationary-noisy"},
    "n_neighbors": {"description": "LOF neighborhood size", "depends_on": "series length / window",
                    "suggest": "~sqrt(window_size); 10-50 typical", "range": [2, 200]},
    "window_size": {"description": "subsequence length for windowed AD",
                    "depends_on": "dominant_period", "suggest": "~1x dominant_period", "range": [4, 1024]},
    "n_clusters": {"description": "number of clusters", "depends_on": "elbow/silhouette over the set",
                   "suggest": "choose by silhouette; start 2-8", "range": [2, 50]},
    "coverage": {"description": "conformal interval coverage", "suggest": "0.9 default; 0.8/0.95 alt",
                 "range": [0.5, 0.99]},
}


def param_schema(card: dict) -> dict:
    """Schema the agent reasons over: introspected params + merged hints (card hints override)."""
    info = introspect(card["sktime_class"])
    hints = {**DEFAULT_HINTS, **(card.get("param_hints") or {})}
    for name, meta in info["params"].items():
        if name in hints:
            meta["hint"] = hints[name]
    info["model_id"] = card.get("model_id")
    return info


def validate_params(card: dict, params: Optional[dict]) -> dict:
    """Check agent-chosen params against the schema before resolve; produce a param_audit."""
    info = introspect(card["sktime_class"])
    known = set(info["params"])
    hints = {**DEFAULT_HINTS, **(card.get("param_hints") or {})}
    issues, audit = [], {}
    for k, v in (params or {}).items():
        if k not in known:
            issues.append(f"unknown param '{k}' (valid: {sorted(known)[:8]}…)")
            continue
        h = hints.get(k, {})
        if "choices" in h and v not in h["choices"]:
            issues.append(f"{k}={v!r} not in choices {h['choices']}")
        if "range" in h and isinstance(v, (int, float)):
            lo, hi = h["range"]
            audit[k] = {"value": v, "in_range": bool(lo <= v <= hi), "range": h["range"]}
            if not (lo <= v <= hi):
                issues.append(f"{k}={v} out of range {h['range']}")
    return {"ok": not issues, "issues": issues, "param_audit": audit, "known_params": sorted(known)}
