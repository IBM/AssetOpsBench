"""Resolve catalog model cards into sktime estimators.

Cards point at `sktime_class` plus JSON params; `_build` also realizes nested estimator specs using
`{"_target_": "...", "params": {...}}`. The CouchDB catalog extends sktime's registry with remote,
fine-tuned, lineage, and benchmark metadata.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

# task_id  ->  sktime scitype (the standardization rides on sktime's taxonomy)
TASK_TO_SCITYPE = {
    "tsfm_forecasting": "forecaster",  # incl. global_forecaster (foundation models)
    "tsfm_regression": "regressor",
    "tsfm_classification": "classifier",
    "tsfm_anomaly_detection": "detector",  # sktime detection scitype (anomaly/segmentation)
    "tsfm_imputation": "transformer",  # sktime Imputer is a transformer
    "tsfm_clustering": "clusterer",
    "tsfm_similarity_search": "transformer",  # feature/embedding transformer + distances
    "tsfm_evaluation": "metric",  # metric + splitter (sktime.evaluate)
}

# Marker key: a param value that is a dict carrying this key is a nested estimator spec.
_TARGET_KEY = "_target_"


def _import_target(path: str):
    """Import a dotted path 'pkg.module.Name' -> the class/callable Name."""
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _build(spec: Any) -> Any:
    """Recursively realize a param value.

    - dict with `_target_`  -> instantiate `Class(**built(params))` (a nested estimator)
    - other dict            -> rebuild values (a plain kwargs dict, e.g. sktime pipeline steps)
    - list/tuple            -> rebuild elements
    - anything else         -> returned as-is
    """
    if isinstance(spec, dict):
        if _TARGET_KEY in spec:
            Cls = _import_target(spec[_TARGET_KEY])
            params = spec.get("params") or {}
            return Cls(**{k: _build(v) for k, v in params.items()})
        return {k: _build(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return type(spec)(_build(v) for v in spec)
    return spec


def resolve(card: dict):
    """Instantiate the sktime estimator a catalog card points at.

    Card `params` are realized with `_build`, so any nested `_target_` estimator specs
    (e.g. the PyOD adapter's `estimator`) are constructed before the outer estimator.
    """
    path = card.get("sktime_class")
    if not path:
        raise ValueError(f"card '{card.get('model_id')}' has no sktime_class")
    Est = _import_target(path)
    params = {k: _build(v) for k, v in (card.get("params") or {}).items()}
    return Est(**params)


def discover(scitype: str, filter_tags: Optional[dict] = None) -> List[str]:
    """sktime registry as live model discovery (installed estimators)."""
    from sktime.registry import all_estimators

    ests = all_estimators(estimator_types=scitype, filter_tags=filter_tags)
    return [n for n, _ in ests]


# foundation-model module/name keywords → pretrained, zero-shot-capable
_FM_KEYS = (
    "tinytime",
    "ttm",
    "chronos",
    "moirai",
    "timesfm",
    "moment",
    "timemoe",
    "patchtst",
    "lagllama",
    "hftransformers",
    "tspulse",
)

# params that signal an *opt-in* fine-tune on a foundation model
_FT_KEYS = (
    "num_train_epochs",
    "fit_strategy",
    "trainer",
    "fine_tune",
    "finetune",
    "lr",
)


def foundation_forecasters() -> List[str]:
    return [n for n in discover("forecaster") if any(k in n.lower() for k in _FM_KEYS)]


def is_foundation(card: dict) -> bool:
    cls = (card.get("sktime_class") or "").lower()
    return any(k in cls for k in _FM_KEYS)


def training_regime(card: dict) -> str:
    """How much training a card needs: zero_shot | fit_on_series | fine_tune.

    The agent reads this to pick the cheapest viable option. A pretrained foundation model is
    zero_shot by default (fit() only loads weights + sets context; it does NOT train on the
    target) unless the recipe passes fine-tune params, which makes it fine_tune. Everything
    else (AutoARIMA/ETS/Theta/reduction) is fit_on_series: cheap parameter estimation on the
    series' own history, no separate training set. An explicit card `training_regime` wins.
    """
    explicit = card.get("training_regime")
    if explicit:
        return explicit
    if is_foundation(card):
        params = card.get("params") or {}
        return "fine_tune" if any(k in params for k in _FT_KEYS) else "zero_shot"
    return "fit_on_series"
