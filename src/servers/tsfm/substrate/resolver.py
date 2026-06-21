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


def resolve(card: dict):
    """Instantiate the sktime estimator a catalog card points at."""
    path = card.get("sktime_class")
    if not path:
        raise ValueError(f"card '{card.get('model_id')}' has no sktime_class")
    module, cls = path.rsplit(".", 1)
    Est = getattr(importlib.import_module(module), cls)
    return Est(**(card.get("params") or {}))


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
