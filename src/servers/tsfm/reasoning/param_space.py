"""param_space.py: per-model parameter schema + reasoning + validation.

Every model has its own parameters whose VALUES must be reasoned (context_length, sp,
n_neighbors, n_clusters, strategy, …). A card therefore exposes a parameter schema:
  - auto-introspected from the sktime class (names, defaults, types, example sets), plus
  - curated `param_hints` per parameter: a one-line description, what data evidence it
    `depends_on`, a `suggest` rule, and an allowed `range`/`choices`.

The agent reads the schema + `profile_series` evidence, REASONS a value for each parameter,
and fills the recipe's `params`. Recipe-block params (finetune / anomaly) are validated against
these hints via validate_block; invalid estimator constructor params are rejected by sktime at
build time; the scorer grades the rest.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Dict, Optional


def introspect(sktime_class: str) -> dict:
    """Constructor params (name → default/required/type) + sktime example param sets."""
    module, cls = sktime_class.rsplit(".", 1)
    Est = getattr(importlib.import_module(module), cls)
    sig = inspect.signature(Est.__init__)
    _empty = inspect.Parameter.empty
    params: Dict[str, dict] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        params[name] = {
            "default": (None if p.default is _empty else _jsonable(p.default)),
            "required": p.default is _empty,
            "type": (None if p.annotation is _empty else _typename(p.annotation)),
        }
    examples = None
    try:
        examples = Est.get_test_params()  # sktime: valid example configs
    except Exception:
        pass
    return {"sktime_class": sktime_class, "params": params, "examples": examples}


def _typename(a):
    return getattr(a, "__name__", str(a))


def _jsonable(v):
    return (
        v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)
    )


# curated reasoning hints for common TS params: what evidence drives the value
DEFAULT_HINTS = {
    "context_length": {
        "description": "input/look-back window the model sees",
        "depends_on": "dominant_period (seasonality)",
        "suggest": ">= 2x dominant_period; must cover the seasonal cycle",
        "range": [8, 2048],
    },
    "prediction_length": {
        "description": "forecast horizon",
        "depends_on": "the request",
        "suggest": "match the horizon named in the task",
        "range": [1, 1024],
    },
    "sp": {
        "description": "seasonal periodicity",
        "depends_on": "dominant_period",
        "suggest": "= dominant_period from profile_series",
        "range": [1, 1024],
    },
    "window_length": {
        "description": "rolling/reduction window",
        "depends_on": "dominant_period",
        "suggest": "~1-2x dominant_period",
        "range": [2, 1024],
    },
    "strategy": {
        "description": "naive strategy",
        "choices": ["last", "mean", "drift"],
        "suggest": "drift if trending, last if persistent, mean if stationary-noisy",
    },
    "n_neighbors": {
        "description": "LOF neighborhood size",
        "depends_on": "series length / window",
        "suggest": "~sqrt(window_size); 10-50 typical",
        "range": [2, 200],
    },
    "window_size": {
        "description": "subsequence length for windowed AD",
        "depends_on": "dominant_period",
        "suggest": "~1x dominant_period",
        "range": [4, 1024],
    },
    "n_clusters": {
        "description": "number of clusters",
        "depends_on": "elbow/silhouette over the set",
        "suggest": "choose by silhouette; start 2-8",
        "range": [2, 50],
    },
    "coverage": {
        "description": "conformal interval coverage",
        "suggest": "0.9 default; 0.8/0.95 alt",
        "range": [0.5, 0.99],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Recipe-block hints: for the run-time algorithm choices that aren't constructor
# params of a single estimator: finetune (training_config) and anomaly (conformal AD).
# These mirror the legacy TTM `_ttm_main_config` and the conformal-AD wrapper knobs, made
# explicit + agent-reasoned. The recipe carries only overrides; defaults fill the rest.
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_HINTS = {
    "n_finetune": {
        "description": "few-shot train size: fraction (<=1) or count (>1)",
        "depends_on": "available history",
        "suggest": "0.05 (5%) for few-shot",
        "range": [0.0, 1e6],
    },
    "n_calibration": {
        "description": "calibration split: fraction/count",
        "suggest": "0 unless conformal",
        "range": [0.0, 1e6],
    },
    "n_test": {
        "description": "held-out test split: fraction/count",
        "suggest": "0.05",
        "range": [0.0, 1e6],
    },
    "lr": {
        "description": "learning rate",
        "suggest": "0 = auto (lr_finder); else 1e-4..1e-2",
        "range": [0.0, 1.0],
    },
    "epochs": {
        "description": "fine-tune epochs",
        "suggest": "few-shot: 1-10",
        "range": [1, 200],
    },
    "batch_size": {
        "description": "mini-batch size",
        "suggest": "32; lower if OOM",
        "range": [1, 1024],
    },
    "patch_length": {
        "description": "TTM patch length",
        "depends_on": "context_length",
        "range": [1, 512],
    },
    "head_dropout": {
        "description": "forecast-head dropout",
        "suggest": "0.7 default regularization",
        "range": [0.0, 0.9],
    },
    "backbone_frozen": {
        "description": "freeze backbone (linear-probe)",
        "choices": [True, False],
        "suggest": "freeze for very small data",
    },
    "decoder_mode": {
        "description": "TTM decoder mixing",
        "choices": ["mix_channel", "common_channel"],
    },
    "scaling": {
        "description": "per-channel scaling",
        "choices": ["", "standard"],
        "suggest": "standard normalizes inputs",
    },
    "p_validation": {
        "description": "validation fraction of the few-shot set",
        "range": [0.0, 0.5],
        "suggest": "0.1",
    },
    "es_patience": {
        "description": "early-stopping patience (epochs)",
        "range": [1, 100],
    },
    "es_th": {"description": "early-stopping min-delta", "range": [0.0, 1.0]},
    "epochs_warmup": {"description": "scheduler warmup epochs", "range": [0, 50]},
    "scheduler": {"description": "LR scheduler", "choices": ["OneCycleLR", "cosine"]},
    "optim": {"description": "optimizer", "choices": ["AdamW"]},
    "num_workers": {"description": "dataloader workers", "range": [0, 32]},
    "seed": {"description": "random seed", "range": [0, 2_147_483_647]},
}

ANOMALY_HINTS = {
    "ad_model_type": {
        "description": "conformal AD variant",
        "choices": ["timeseries_conformal", "timeseries_conformal_adaptive"],
        "suggest": "adaptive for non-stationary signals",
    },
    "false_alarm": {
        "description": "target false-alarm rate (= 1 - coverage)",
        "suggest": "0.05 → 95% coverage",
        "range": [0.001, 0.5],
    },
    "n_calibration": {
        "description": "fraction of data for conformal calibration",
        "suggest": "0.2",
        "range": [0.0, 1.0],
    },
    "threshold_function": {
        "description": "threshold rule",
        "choices": ["weighting", "static"],
        "suggest": "weighting for the adaptive variant",
    },
    "window_size": {
        "description": "calibration window length",
        "depends_on": "dominant_period",
        "suggest": "≈ recent-regime length; null = all calibration",
        "range": [4, 100000],
    },
    "nonconformity_score": {
        "description": "score function",
        "choices": ["absolute_error"],
    },
    "decay_param": {
        "description": "exponential weighting decay (adaptive)",
        "range": [0.0, 1.0],
    },
    "task": {
        "description": "fit a new AD model or run inference",
        "choices": ["fit", "inference"],
    },
    "impute": {
        "description": "how to handle missing values before a classical (non-foundation) "
        "detector; omit to let the model raise on NaN (foundation detectors "
        "such as tspulse accept NaN natively)",
        "choices": ["interpolate", "drop", "zero"],
        "suggest": "interpolate for sensor gaps; drop to avoid fabricating values; "
        "zero only when 0 is meaningful",
    },
}

BLOCK_HINTS = {"finetune": FINETUNE_HINTS, "anomaly": ANOMALY_HINTS}


def validate_block(block: str, params: Optional[dict]) -> dict:
    """Validate a recipe-block param dict (finetune/anomaly) against its hints; produce an audit.
    Free-form dict (no sktime constructor): the hint set defines the known params."""
    hints = BLOCK_HINTS.get(block, {})
    known = set(hints)
    issues, audit = [], {}
    for k, v in (params or {}).items():
        if k not in known:
            issues.append(f"unknown {block} param '{k}' (valid: {sorted(known)[:8]}…)")
            continue
        h = hints[k]
        if "choices" in h and v not in h["choices"]:
            issues.append(f"{block}.{k}={v!r} not in choices {h['choices']}")
        if "range" in h and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = h["range"]
            audit[k] = {
                "value": v,
                "in_range": bool(lo <= v <= hi),
                "range": h["range"],
            }
            if not (lo <= v <= hi):
                issues.append(f"{block}.{k}={v} out of range {h['range']}")
    return {
        "ok": not issues,
        "issues": issues,
        "param_audit": audit,
        "known_params": sorted(known),
    }


def param_schema(card: dict) -> dict:
    """Schema the agent reasons over: introspected params + merged hints (card hints override).
    Estimator params themselves are validated by sktime at construction time."""
    sk = card.get("sktime_class")
    if not sk:
        raise ValueError(
            f"card '{card.get('model_id')}' has no sktime_class "
            "(toolkit/checkpoint model); no constructor param schema"
        )
    info = introspect(sk)
    hints = {**DEFAULT_HINTS, **(card.get("param_hints") or {})}
    for name, meta in info["params"].items():
        if name in hints:
            meta["hint"] = hints[name]
    info["model_id"] = card.get("model_id")
    return info