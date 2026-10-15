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


# Only these classes may be imported from a dotted path. Paths reach this function from agent
# input (inline sktime_class, nested _target_, catalog cards the agent can rewrite), so an
# open import ran any callable, e.g. subprocess.Popen. Operators extend the list with
# TSFM_EXTRA_ESTIMATOR_CLASSES (comma-separated exact paths).
ALLOWED_CLASSES = frozenset({
    "sktime.forecasting.ttm.TinyTimeMixerForecaster",
    "sktime.forecasting.chronos.ChronosForecaster",
    "sktime.forecasting.timesfm.TimesFMForecaster",
    "sktime.forecasting.timesfm2.TimesFM2Forecaster",
    "sktime.forecasting.moirai.MOIRAIForecaster",
    "sktime.forecasting.moirai2.Moirai2Forecaster",
    "sktime.forecasting.toto2.Toto2Forecaster",
    "sktime.forecasting.kronos.KronosForecaster",
    "sktime.forecasting.momentfm.MomentFMForecaster",
    "sktime.forecasting.timemoe.TimeMoEForecaster",
    "sktime.forecasting.patch_tst.PatchTSTForecaster",
    "sktime.forecasting.patch_tsmixer.PatchTSMixerForecaster",
    "sktime.forecasting.naive.NaiveForecaster",
    "sktime.forecasting.theta.ThetaForecaster",
    "sktime.forecasting.arima.ARIMA",
    "sktime.forecasting.arima.AutoARIMA",
    "sktime.forecasting.conformal.ConformalIntervals",
    "sktime.detection.tspulse.TSPulseAnomalyDetector",
    "sktime.detection.hampel.HampelDetector",
    "sktime.detection.lof.SubLOF",
    "sktime.detection.adapters._pyod.PyODDetector",
    "sktime.detection.dummy._zero_an.ZeroAnomalies",
    "pyod.models.iforest.IForest",
    "sktime.classification.foundation_models.tspulse.TSPulseClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "sktime.clustering.k_means.TimeSeriesKMeans",
    "sktime.clustering.k_medoids.TimeSeriesKMedoids",
    "sktime.clustering.dbscan.TimeSeriesDBSCAN",
    "sktime.clustering.kernel_k_means.TimeSeriesKernelKMeans",
    "sktime.clustering.agglomerative.TimeSeriesAgglomerativeClustering",
    # forecasting `transforms` (TransformedTargetForecaster steps)
    "sktime.transformations.series.detrend.Detrender",
    "sktime.transformations.series.detrend.Deseasonalizer",
    "sktime.transformations.series.difference.Differencer",
    "sktime.transformations.series.boxcox.LogTransformer",
    "sktime.transformations.series.boxcox.BoxCoxTransformer",
    "sktime.transformations.series.exponent.ExponentTransformer",
    "sktime.transformations.series.impute.Imputer",
    "sktime.transformations.series.adapt.TabularToSeriesAdaptor",
    "sktime.forecasting.trend.PolynomialTrendForecaster",  # Detrender's default forecaster
    "sklearn.preprocessing.StandardScaler",
    "sklearn.preprocessing.MinMaxScaler",
    "sktime.transformations.series.exponent.SqrtTransformer",
    "sktime.transformations.series.detrend.ConditionalDeseasonalizer",
    "sktime.transformations.series.outlier_detection.HampelFilter",
    "sktime.forecasting.trend.TrendForecaster",
    "sklearn.preprocessing.RobustScaler",
    # run_tabular_recipe transforms (panel -> features)
    "sktime.transformations.series.summarize.SummaryTransformer",
    "sktime.transformations.panel.catch22.Catch22",
    "sktime.transformations.panel.rocket.MiniRocket",
    "sktime.transformations.panel.rocket.MiniRocketMultivariate",
    "sktime.transformations.panel.rocket.Rocket",
    # run_tabular_recipe estimators: tsfm_regression / tsfm_classification / tsfm_clustering
    "sklearn.ensemble.RandomForestRegressor",
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.linear_model.Ridge",
    "sklearn.cluster.KMeans",
    "sklearn.ensemble.ExtraTreesRegressor",
    "sklearn.linear_model.LinearRegression",
    "sklearn.linear_model.Lasso",
    "sklearn.neighbors.KNeighborsRegressor",
    "sklearn.ensemble.ExtraTreesClassifier",
    "sklearn.ensemble.GradientBoostingClassifier",
    "sklearn.linear_model.LogisticRegression",
    "sklearn.neighbors.KNeighborsClassifier",
})


def _import_target(path: str):
    """Import an allowlisted dotted path 'pkg.module.Name' -> the class Name.

    The check runs before the import, so a refused path never loads its module.
    """
    import os

    extra = {p.strip() for p in os.environ.get("TSFM_EXTRA_ESTIMATOR_CLASSES", "").split(",")}
    if path not in ALLOWED_CLASSES and path not in extra - {""}:
        raise ValueError(f"'{path}' is not an allowed estimator class; use a catalog model_id")
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
    if isinstance(params.get("training_args"), dict):
        params["training_args"] = _safe_training_args(params["training_args"])
    return Est(**params)


# HF TrainingArguments keys that name a path or an outside service. output_dir is forced to a
# server-chosen folder; the rest are refused, so a fine-tune writes (and pickles) only there.
_TRAINING_ARGS_REFUSED = frozenset({
    "logging_dir", "resume_from_checkpoint", "deepspeed", "fsdp_config", "accelerator_config",
    "push_to_hub", "hub_model_id", "hub_token", "hub_strategy", "hub_private_repo",
    "hub_always_push", "run_name",
})


def _safe_training_args(ta: dict) -> dict:
    import os
    import uuid

    from ..io import refs

    bad = sorted(k for k in ta if k in _TRAINING_ARGS_REFUSED or k.startswith("hub_"))
    if bad:
        raise ValueError(f"training_args may not set {bad}")
    out = dict(ta)
    out["output_dir"] = os.path.join(refs._ensure_workdir(), "trainer", uuid.uuid4().hex[:8])
    out["report_to"] = "none"
    return out


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
