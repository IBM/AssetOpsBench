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

# Params that signal an *opt-in* fine-tune on a foundation model. NOTE these are matched by
# PRESENCE, so they must only list keys whose mere presence means "train" regardless of value.
# `fit_strategy` is deliberately NOT here: it is tri-valued (zero-shot|minimal|full) and is handled
# by value in _regime_from_params below.
_FT_KEYS = (
    "num_train_epochs",
    "trainer",
    "fine_tune",
    "finetune",
    "lr",
)

# fit_strategy values (and equivalents) that mean "use the weights as they are, do not train"
_ZERO_SHOT_VALUES = {"zero-shot", "zero_shot", "zeroshot", "none", "no", "off"}

# Estimator constructor params whose presence means a training loop was configured. Estimators are
# not consistent here: sktime's TTM/PatchTST use `fit_strategy` + HF `training_args`, while the
# MomentFM family uses `epochs` / `max_lr`. Matching only the TTM spelling silently misses the rest.
_TRAIN_CONFIG_KEYS = ("training_args", "epochs", "max_lr", "max_epochs", "learning_rate")


def foundation_forecasters() -> List[str]:
    return [n for n in discover("forecaster") if any(k in n.lower() for k in _FM_KEYS)]


def is_foundation(card: dict) -> bool:
    cls = (card.get("sktime_class") or "").lower()
    return any(k in cls for k in _FM_KEYS)


def _estimator_default_regime(sktime_class: str) -> str:
    """What a foundation estimator does on fit() when the card configures nothing.

    "Pretrained" does NOT imply "zero-shot on fit". sktime's defaults vary and several of them
    TRAIN:

        TinyTimeMixerForecaster   fit_strategy="minimal"  -> fine-tunes a parameter subset
        PatchTSTForecaster        fit_strategy="full"     -> full fine-tune
        MomentFMForecaster        epochs=1, freeze_head=False -> trains a head
        ChronosForecaster         (no training params)    -> genuinely zero-shot

    So the default regime has to be read off the estimator, not assumed. Falls back to zero_shot
    only when the class exposes no training knob at all, or cannot be inspected.
    """
    import inspect

    try:
        target = _import_target(sktime_class)
        sig = inspect.signature(target.__init__).parameters
    except Exception:
        return "zero_shot"  # uninspectable: keep the old optimistic assumption

    if "fit_strategy" in sig:
        default = sig["fit_strategy"].default
        if default is inspect.Parameter.empty or default is None:
            return "zero_shot"
        return (
            "zero_shot"
            if str(default).lower() in _ZERO_SHOT_VALUES
            else "fine_tune"  # e.g. TTM "minimal", PatchTST "full"
        )

    # MomentFM-style: an epochs default > 0 with a trainable head means fit() trains.
    epochs = sig["epochs"].default if "epochs" in sig else None
    if isinstance(epochs, int) and epochs > 0:
        frozen = sig["freeze_head"].default if "freeze_head" in sig else True
        if frozen is False:
            return "fine_tune"

    return "zero_shot"


def _regime_from_params(params: dict, sktime_class: str) -> str:
    """Regime implied by a foundation card's params, honouring VALUES and estimator defaults."""
    # 1. An explicit fit_strategy is the estimator's own switch. Read its value.
    if "fit_strategy" in params:
        return (
            "zero_shot"
            if str(params["fit_strategy"]).lower() in _ZERO_SHOT_VALUES
            else "fine_tune"
        )
    # 2. A training config was supplied (HF training_args, epochs, lr, ...) -> training.
    if any(params.get(k) for k in _TRAIN_CONFIG_KEYS):
        return "fine_tune"
    if any(k in params for k in _FT_KEYS):
        return "fine_tune"
    # 3. Nothing configured: whatever the estimator does by default.
    return _estimator_default_regime(sktime_class)


def training_regime(card: dict) -> str:
    """How much training a card needs: zero_shot | fit_on_series | fine_tune.

    The agent reads this to pick the cheapest viable option, and `run_recipe` uses it to choose
    between a single zero-shot holdout and the expanding-window refit loop. So it has to be right.

    Rules, in order:
      1. An explicit card `training_regime` wins. Always. It is the only fully reliable signal and
         foundation cards should set it.
      2. Non-foundation cards are fit_on_series: cheap parameter estimation on the series' own
         history (AutoARIMA/ETS/Theta/reduction).
      3. Foundation cards: an explicit `fit_strategy` is read BY VALUE ("zero-shot" -> zero_shot,
         "minimal"/"full" -> fine_tune); a supplied training config (`training_args`, `epochs`,
         `max_lr`, ...) means fine_tune; otherwise the ESTIMATOR'S OWN DEFAULT decides, because
         several pretrained estimators fine-tune when handed nothing (TTM "minimal",
         PatchTST "full", MomentFM epochs=1).

    Note `provenance` is history and does not enter into this: a finetuned checkpoint being SERVED
    is zero_shot (load the weights, predict), which is why register_finetuned pins the regime.
    """
    explicit = card.get("training_regime")
    if explicit:
        return explicit
    if not is_foundation(card):
        return "fit_on_series"
    return _regime_from_params(card.get("params") or {}, card.get("sktime_class") or "")
