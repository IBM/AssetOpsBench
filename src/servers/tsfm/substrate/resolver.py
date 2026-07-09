"""sktime_resolver.py — the model/feature store on top of sktime (the substrate).

Key decision: do NOT hand-roll an estimator/transform contract. Adopt sktime's scitype +
tag + registry system. A catalog card is just a *pointer* to an sktime estimator class + its
constructor params + tags; resolving a card = import & instantiate; running it = the scitype's
verb (forecaster.predict / classifier.predict / detector.predict / transformer.transform /
clusterer.predict). Heterogeneous foundation-model code paths disappear: every TSFM (TTM,
Chronos, MOIRAI, TimesFM, MOMENT, TimeMoE, PatchTST, …) is a `BaseForecaster` in sktime.

Our catalog (CouchDB) is a *superset* of sktime's in-memory registry: it also holds
not-installed / remote / fine-tuned models with provenance, lineage, metrics — and is
agent-queryable and state-exportable (#394). sktime supplies fit/predict/pipeline/splitter/
metric; we supply catalog + selection (T-Daub) + reasoning + persistence + MCP.

Nested estimators (#pyod_iforest): some sktime estimators take *another estimator instance*
as a constructor arg (e.g. the PyOD adapter's `estimator=IsolationForest()`). JSON can't hold a
live object, so a param value may instead be a nested spec dict:

    {"_target_": "pyod.models.iforest.IForest", "params": {"contamination": 0.1}}

`_build` recursively turns any such spec into `Class(**params)` before the outer estimator is
instantiated. Plain params (and dicts without `_target_`) pass through unchanged, so this is
fully backward-compatible with existing cards.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

# task_id  ->  sktime scitype (the standardization rides on sktime's taxonomy)
TASK_TO_SCITYPE = {
    "tsfm_forecasting": "forecaster",          # incl. global_forecaster (foundation models)
    "tsfm_regression": "regressor",
    "tsfm_classification": "classifier",
    "tsfm_anomaly_detection": "detector",      # sktime detection scitype (anomaly/segmentation)
    "tsfm_imputation": "transformer",          # sktime Imputer is a transformer
    "tsfm_clustering": "clusterer",
    "tsfm_similarity_search": "transformer",   # feature/embedding transformer + distances
    "tsfm_evaluation": "metric",               # metric + splitter (sktime.evaluate)
}

# scitype -> the verb to call after fit
_VERB = {"forecaster": "predict", "regressor": "predict", "classifier": "predict",
         "clusterer": "predict", "detector": "predict", "transformer": "transform",
         "metric": None}

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
_FM_KEYS = ("tinytime", "ttm", "chronos", "moirai", "timesfm", "moment", "timemoe",
            "patchtst", "lagllama", "hftransformers", "tspulse")

# params that signal an *opt-in* fine-tune on a foundation model
_FT_KEYS = ("num_train_epochs", "fit_strategy", "trainer", "fine_tune", "finetune", "lr")


def foundation_forecasters() -> List[str]:
    return [n for n in discover("forecaster") if any(k in n.lower() for k in _FM_KEYS)]


def is_foundation(card: dict) -> bool:
    cls = (card.get("sktime_class") or "").lower()
    return any(k in cls for k in _FM_KEYS)


def training_regime(card: dict) -> str:
    """How much training a card needs: zero_shot | fit_on_series | fine_tune.

    The agent reads this to pick the cheapest viable option. A pretrained foundation model is
    zero_shot by default (fit() only loads weights + sets context; it does NOT train on the
    target) — unless the recipe passes fine-tune params, which makes it fine_tune. Everything
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


def run(card: dict, task_id: str, *, y=None, X=None, fh=None):
    """Fit + run an sktime estimator via its scitype verb. Returns the native sktime output."""
    scitype = TASK_TO_SCITYPE.get(task_id)
    est = resolve(card)
    verb = _VERB.get(scitype)
    if scitype == "forecaster":
        est.fit(y, X=X, fh=fh) if X is not None else est.fit(y, fh=fh)
        return est.predict(fh=fh)
    if scitype in ("classifier", "regressor", "clusterer"):
        est.fit(X, y); return getattr(est, verb)(X)
    if scitype == "detector":
        est.fit(X if X is not None else y); return est.predict(X if X is not None else y)
    if scitype == "transformer":
        est.fit(y if y is not None else X)
        return est.transform(y if y is not None else X)
    raise ValueError(f"no runner for scitype '{scitype}'")