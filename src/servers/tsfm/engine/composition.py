"""composition.py — the CORE capability: agentic mix-and-match + ensemble + iterate.

The loop:
  1. discover_components()  → the agent sees every transform / model / combiner / metric /
     splitter it can compose (catalog ∪ sktime registry).
  2. the agent authors a RECIPE (a declarative spec: transforms + single model OR an ensemble
     {mean|median|weighted|stack} of members + an eval protocol).
  3. run_recipe()  → the server COMPILES the recipe to an sktime forecaster (ensembles =
     EnsembleForecaster/StackingForecaster; transforms = TransformedTargetForecaster with
     automatic inverse), BACKTESTS it (sktime.evaluate + splitter), produces the final
     forecast, and returns rich DIAGNOSTICS (ensemble score + per-member scores + weights).
  4. the agent inspects the diagnostics and submits a REVISED recipe (drop a weak member,
     reweight, change lookback) → another round. Each run is persisted with a parent link, so
     the refinement trajectory is state (GIFT-Eval-style ensemble search, agent-driven).

sktime is the substrate; this module is the thin composition/iterate engine on top.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RUNS = "tsfm_runs"


def _impute(y: pd.Series, how: str) -> pd.Series:
    """Apply an EXPLICIT imputation strategy from recipe['impute'].
    interpolate = linear fill of interior gaps + nearest at edges (0.0 if all-NaN);
    drop        = remove NaN rows (indices are remapped to original positions by the caller);
    zero        = fill NaN with 0.0."""
    if how == "interpolate":
        y = y.astype(float).interpolate(method="linear", limit_direction="both")
        return y.ffill().bfill().fillna(0.0)
    if how == "drop":
        return y.dropna()
    if how == "zero":
        return y.astype(float).fillna(0.0)
    raise ValueError(f"unknown impute strategy '{how}' (use interpolate|drop|zero)")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _metric(name):
    from sktime.performance_metrics.forecasting import (
        MeanAbsolutePercentageError,
        MeanAbsoluteError,
        MeanAbsoluteScaledError,
    )

    return {
        "smape": MeanAbsolutePercentageError(symmetric=True),
        "mape": MeanAbsolutePercentageError(),
        "mae": MeanAbsoluteError(),
        "mase": MeanAbsoluteScaledError(),
    }.get(name, MeanAbsolutePercentageError(symmetric=True))


def _resolve_estimator(spec: dict, store=None):
    """estimator spec → (name, sktime_estimator). spec: {name?, model_id?|sktime_class, params?}."""
    from ..substrate import resolver as R

    name = (
        spec.get("name")
        or spec.get("model_id")
        or spec.get("sktime_class", "est").split(".")[-1]
    )
    if spec.get("model_id") and store is not None:
        from ..stores import model_store
        card = model_store.get_model(store, spec["model_id"])
        if not card:
            raise ValueError(f"model '{spec['model_id']}' not in catalog")
        merged = {**card, "params": {**(card.get("params") or {}), **(spec.get("params") or {})}}
        return name, R.resolve(merged)
    return name, R.resolve(spec)

def build_forecaster(recipe: dict, store=None):
    """Compile a recipe into a single sktime forecaster (ensemble/transform-aware)."""
    from sktime.forecasting.compose import EnsembleForecaster

    # 1) the core forecaster: single or ensemble
    if "ensemble" in recipe:
        ens = recipe["ensemble"]
        members = [_resolve_estimator(m, store) for m in ens["members"]]
        combine = ens.get("combine", "mean")
        if combine == "weighted":
            base = EnsembleForecaster(members, weights=ens.get("weights"))
        elif combine == "stack":
            from sktime.forecasting.compose import StackingForecaster

            base = StackingForecaster(members)
        else:  # mean | median | min | max
            base = EnsembleForecaster(members, aggfunc=combine)
    else:
        _, base = _resolve_estimator(recipe["estimator"], store)

    # 2) optional transforms with automatic inverse (AutoAI-TS reverse-order, via sktime)
    transforms = recipe.get("transforms") or []
    if transforms:
        from sktime.forecasting.compose import TransformedTargetForecaster
        from ..substrate import resolver as R

        steps = []
        for i, t in enumerate(transforms):
            steps.append((t.get("name", f"t{i}"), R.resolve(t)))
        steps.append(("forecaster", base))
        base = TransformedTargetForecaster(steps)

    # 3) optional conformal / probabilistic wrapper — adds calibrated prediction intervals to
    #    ANY recipe (single, ensemble, or transformed) via sktime ConformalIntervals.
    if recipe.get("conformal"):
        from sktime.forecasting.conformal import ConformalIntervals

        base = ConformalIntervals(base)
    return base


def _backtest(forecaster, y, recipe):
    from sktime.split import ExpandingWindowSplitter
    from sktime.forecasting.model_evaluation import evaluate

    ev = recipe.get("eval", {})
    fh = recipe.get("fh", [1, 2, 3, 4, 5])
    iw = ev.get("initial_window", max(len(y) // 2, 2 * len(fh)))
    step = ev.get("step", max(len(fh), 1))
    metric = _metric((ev.get("metrics") or ["smape"])[0])
    cv = ExpandingWindowSplitter(initial_window=iw, step_length=step, fh=fh)
    res = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=metric)
    col = [c for c in res.columns if c.startswith("test_")][0]
    return float(res[col].mean()), metric.__class__.__name__, len(res)


def _backtest_zero_shot(forecaster, y, recipe):
    """Fast path for pretrained models: NO retraining loop. One rolling-origin holdout —
    set the context = all-but-last-h, run inference once, score the held-out tail. Because a
    zero-shot model isn't trained, re-fitting it across many expanding folds only repeats
    inference; a single holdout is the honest, cheap estimate."""
    fh = recipe.get("fh", [1, 2, 3, 4, 5])
    h = len(fh)
    y_train, y_test = y.iloc[:-h], y.iloc[-h:]
    metric = _metric((recipe.get("eval", {}).get("metrics") or ["smape"])[0])
    forecaster.fit(y_train, fh=fh)  # fit = load weights + set context
    y_pred = np.asarray(forecaster.predict())[:h]
    yt = np.asarray(y_test)[: len(y_pred)]
    try:
        score = float(metric(yt, y_pred, y_train=np.asarray(y_train)))  # MASE etc.
    except TypeError:
        score = float(metric(yt, y_pred))
    return score, metric.__class__.__name__, 1


def _recipe_regime(recipe, store) -> str:
    """Cheapest regime that covers the whole recipe: zero_shot only if every estimator is
    pretrained-zero-shot and no fine-tune is requested; else fit_on_series (or fine_tune).
    """
    from ..substrate import resolver as R

    if recipe.get("finetune"):
        return "fine_tune"
    specs = (
        recipe["ensemble"]["members"] if "ensemble" in recipe else [recipe["estimator"]]
    )
    regimes = set()
    for s in specs:
        card = s
        if s.get("model_id") and store is not None:
            from ..stores import model_store

            card = model_store.get_model(store, s["model_id"]) or s
        merged = {
            **card,
            "params": {**(card.get("params") or {}), **(s.get("params") or {})},
        }
        regimes.add(R.training_regime(merged))
    if "fine_tune" in regimes:
        return "fine_tune"
    return "zero_shot" if regimes == {"zero_shot"} else "fit_on_series"


def _validate_blocks(recipe: dict) -> dict:
    """Validate any finetune/anomaly recipe blocks against param_space hints (non-fatal): the
    audit is recorded so the agent's run-time choices are graded and bad values surface.
    """
    from ..reasoning import param_space as PS

    audit = {}
    for block in ("finetune", "anomaly"):
        if isinstance(recipe.get(block), dict):
            audit[block] = PS.validate_block(block, recipe[block])
    return audit


def run_recipe(
    store,
    y,
    recipe: dict,
    *,
    asset_id: str = "asset",
    parent_run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """Compile → backtest → final forecast → per-member diagnostics → persist run (with lineage).
    Dispatches by recipe['task']: anomaly_detection routes to the detector path (run_anomaly);
    everything else is forecasting."""
    if recipe.get("task") == "tsfm_anomaly_detection":
        return run_anomaly(
            store,
            y,
            recipe,
            asset_id=asset_id,
            parent_run_id=parent_run_id,
            scenario_id=scenario_id,
        )

    # normalize to plain float64 (nullable dtypes keep pd.NA, which crashes int() inside sktime)
    y = pd.Series(np.asarray(y, dtype=float))
    regime = _recipe_regime(recipe, store)
    # missing values: apply an EXPLICIT recipe['impute'] (interpolate|drop|zero); otherwise, if a
    # classical forecaster is asked to fit a gapped series, fail clearly. Foundation/zero-shot
    # models tolerate gaps, so they proceed untouched.
    if recipe.get("impute"):
        y = _impute(y, recipe["impute"])
    elif y.isna().any() and regime != "zero_shot":
        raise ValueError(
            "target series has missing values; set recipe['impute'] to "
            "'interpolate', 'drop', or 'zero' (classical forecasters cannot fit gaps)."
        )
    fc = build_forecaster(recipe, store)
    block_audit = _validate_blocks(recipe)
    _bt = _backtest_zero_shot if regime == "zero_shot" else _backtest
    score, metric_name, folds = _bt(fc, y, recipe)

    # per-member diagnostics (so the agent can drop/reweight) — only for ensembles
    per_member = {}
    if "ensemble" in recipe:
        for m in recipe["ensemble"]["members"]:
            name, est = _resolve_estimator(m, store)
            try:
                per_member[name] = round(_bt(est, y, recipe)[0], 4)
            except Exception as e:
                per_member[name] = f"err:{type(e).__name__}"

    fh = recipe.get("fh", [1, 2, 3, 4, 5])
    fc.fit(y, fh=fh)
    forecast = fc.predict().round(4).tolist()
    intervals = None
    if recipe.get("conformal"):
        cov = (
            recipe["conformal"].get("coverage", 0.9)
            if isinstance(recipe["conformal"], dict)
            else 0.9
        )
        try:
            intervals = fc.predict_interval(coverage=cov).round(4).head(3).to_dict()
        except Exception as e:
            intervals = {"error": str(e)[:80]}

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {
        "_id": run_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "asset_id": asset_id,
        "scenario_id": scenario_id,
        "recipe": recipe,
        "training_regime": regime,
        "trained": regime != "zero_shot",
        "block_audit": block_audit,
        "metric": metric_name,
        "backtest_score": round(score, 4),
        "folds": folds,
        "per_member_score": per_member,
        "forecast_head": forecast[:5],
        "prediction_interval": intervals,
        "created_at": _now(),
    }
    if store is not None:
        store.put(RUNS, rec)
    return {
        "run_id": run_id,
        "task": "tsfm_forecasting",
        "backtest_score": round(score, 4),
        "metric": metric_name,
        "training_regime": regime,
        "trained": regime != "zero_shot",
        "block_audit": block_audit,
        "per_member_score": per_member,
        "forecast_head": forecast[:5],
        "prediction_interval": intervals,
        "parent_run_id": parent_run_id,
        "improved": (
            parent_run_id is not None
            and store is not None
            and score
            < (store.get(RUNS, parent_run_id) or {}).get("backtest_score", 9e9)
        ),
    }


TABULAR_TASKS = {"tsfm_regression", "tsfm_classification", "tsfm_clustering"}


def _as_panel(X):
    """Accept (n, T) univariate or (n, C, T) multivariate → (n, C, T)."""
    X = np.asarray(X, float)
    return X[:, None, :] if X.ndim == 2 else X


def _tile(x, window: Optional[int]) -> np.ndarray:
    """1D series -> (n_windows, window_len). window None/<=0/>=len => a single whole-series window;
    otherwise NON-OVERLAPPING tiles of length `window` (trailing remainder dropped)."""
    x = np.asarray(x, dtype=float)
    if window is None or window <= 0 or window >= len(x):
        return x[None, :]
    n = len(x) // window
    return x[: n * window].reshape(n, window) if n else x[None, :]


def extract_features(channels: Dict[str, Any], extractor_names, window: Optional[int] = None):
    """Apply named FLOps extractors to each channel's windows.
    channels: {column_name -> 1D array}. Returns (columns, matrix) where matrix is
    n_windows x (n_channels * n_extractors); column names are '<channel>.<extractor>' when
    multivariate, else just '<extractor>'. Whole-series => one row."""
    from ..reasoning import feature_selection as FS

    multi = len(channels) > 1
    per = {ch: _tile(x, window) for ch, x in channels.items()}
    nw = min((W.shape[0] for W in per.values()), default=0)
    cols, data = [], []
    for ch, W in per.items():
        W = W[:nw]
        for name in extractor_names:
            fn = FS.EXTRACTORS[name]
            data.append([float(fn(W[i])) for i in range(nw)])
            cols.append(f"{ch}.{name}" if multi else name)
    F = np.nan_to_num(np.column_stack(data)) if data else np.zeros((nw, 0))
    return cols, F

def _lib_features(X, subset=None):
    """Dependency-free 'FeatureUnion': apply the FLOps extractor library per instance/channel."""
    from ..reasoning import feature_selection as FS

    names = [n for n in (subset or FS.EXTRACTORS) if n in FS.EXTRACTORS]
    Xp = _as_panel(X)
    cols, colnames = [], []
    for c in range(Xp.shape[1]):
        for n in names:
            cols.append([FS.EXTRACTORS[n](Xp[i, c]) for i in range(Xp.shape[0])])
            colnames.append(f"c{c}.{n}" if Xp.shape[1] > 1 else n)
    F = np.nan_to_num(np.column_stack(cols)) if cols else np.zeros((len(Xp), 0))
    return F, colnames


def _tabular_features(X, recipe, y=None):
    """Compile the recipe's transforms into a tabular feature matrix (the FeatureUnion).
    A transform spec may be: {extractors:[names]} (our library), {sktime_class:...} (an sktime
    panel transformer, soft-deps permitting), or {flops_select:true} (extract the full library
    then keep columns the multi-config FLOps scorers rank above the rest)."""
    transforms = recipe.get("transforms")
    if not transforms:
        return _lib_features(X)  # default = full library
    parts, names = [], []
    for t in transforms:
        if t.get("flops_select"):
            from ..reasoning import feature_selection as FS

            F, nm = _lib_features(X)
            if y is not None:
                keep = _flops_columns(F, np.asarray(y), nm, t.get("flops_select"))
                idx = [nm.index(k) for k in keep]
                F, nm = F[:, idx], keep
            parts.append(F)
            names += nm
        elif t.get("extractors"):
            F, nm = _lib_features(X, t["extractors"])
            parts.append(F)
            names += nm
        elif t.get("sktime_class"):
            from ..substrate import resolver as R

            tr = R.resolve(t)
            Ft = np.nan_to_num(
                np.asarray(tr.fit_transform(_as_panel(X))).reshape(len(X), -1)
            )
            parts.append(Ft)
            names += [
                f"{t['sktime_class'].split('.')[-1]}_{k}" for k in range(Ft.shape[1])
            ]
    F = np.column_stack(parts) if parts else _lib_features(X)[0]
    return np.nan_to_num(F), names


def _flops_columns(F, y, names, cfg):
    """Reuse the v2 multi-config scorers to select tabular feature COLUMNS by relevance to y."""
    from ..reasoning import feature_selection as FS

    scorers = (cfg.get("scorers") if isinstance(cfg, dict) else None) or [
        "corr",
        "f_test",
        "mutual_info",
        "model",
    ]
    use = [s for s in scorers if s in FS._SCORERS] or ["corr"]
    ranks = np.zeros(len(names))
    for s in use:
        sv = np.asarray(FS._SCORERS[s](F, y, names))
        order = (-sv).argsort()
        rk = np.empty(len(names))
        rk[order] = np.arange(1, len(names) + 1)
        ranks += rk
    mean_rank = ranks / len(use)
    cut = (cfg.get("top_k") if isinstance(cfg, dict) else None) or max(
        1, len(names) // 2
    )
    keep_idx = np.argsort(mean_rank)[:cut]
    return [names[i] for i in sorted(keep_idx)]


def run_tabular_recipe(
    store,
    X,
    recipe: dict,
    *,
    y=None,
    asset_id: str = "asset",
    parent_run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """Series→tabular run path: FeatureUnion(extractors) → estimator, for regression /
    classification / clustering. CV-scored (supervised) or silhouette (clustering). Same recipe
    grammar + persistence + lineage as forecasting; the agent mixes-and-matches features here too.
    """
    from ..substrate import resolver as R
    from sklearn.model_selection import cross_val_score

    task = recipe.get("task", "tsfm_classification")
    F, feat_names = _tabular_features(X, recipe, y)
    regime = (
        _recipe_regime(recipe, store)
        if "estimator" in recipe or "ensemble" in recipe
        else "fit_on_series"
    )
    _, est = _resolve_estimator(recipe["estimator"], store)

    metric, score = None, None
    if task == "tsfm_clustering":
        from sklearn.metrics import silhouette_score

        labels = (
            est.fit_predict(F) if hasattr(est, "fit_predict") else est.fit(F).predict(F)
        )
        try:
            score, metric = round(float(silhouette_score(F, labels)), 4), "silhouette"
        except Exception:
            score, metric = None, "silhouette"
        preds = np.asarray(labels)[:10].tolist()
    else:
        y = np.asarray(y)
        scoring = "accuracy" if task == "tsfm_classification" else "r2"
        nfold = int(
            min(
                5,
                max(
                    2,
                    (
                        np.min(np.bincount(y))
                        if task == "tsfm_classification"
                        else len(y) // 3
                    ),
                ),
            )
        )
        try:
            score = round(
                float(cross_val_score(est, F, y, cv=nfold, scoring=scoring).mean()), 4
            )
        except Exception as e:
            score = f"err:{type(e).__name__}"
        metric = scoring
        est.fit(F, y)
        preds = np.asarray(est.predict(F))[:10].tolist()

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {
        "_id": run_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "task": task,
        "asset_id": asset_id,
        "scenario_id": scenario_id,
        "recipe": recipe,
        "training_regime": regime,
        "n_features": F.shape[1],
        "feature_names": feat_names[:40],
        "metric": metric,
        "cv_score": score,
        "predictions_head": preds,
        "created_at": _now(),
    }
    if store is not None:
        store.put(RUNS, rec)
    return {
        "run_id": run_id,
        "task": task,
        "metric": metric,
        "cv_score": score,
        "n_features": F.shape[1],
        "feature_names": feat_names[:40],
        "predictions_head": preds,
        "training_regime": regime,
        "parent_run_id": parent_run_id,
    }


def _anomaly_labels(pred, n: int):
    """Normalize an sktime detector's predict() to a dense 0/1 label vector + anomaly indices.
    sktime detection returns either anomaly POSITIONS (a DataFrame/array of ilocs) or a dense
    0/1(-1) series — handle both."""
    arr = np.asarray(getattr(pred, "values", pred)).ravel()
    labels = np.zeros(n, dtype=int)
    uniq = set(np.unique(arr).tolist()) if arr.size else set()
    if arr.size == n and uniq.issubset({0, 1, -1}):  # dense labels
        labels = (arr != 0).astype(int)
    else:  # anomaly positions
        idx = arr[(arr >= 0) & (arr < n)].astype(int)
        labels[idx] = 1
    return labels, np.where(labels == 1)[0].tolist()


def _conformal_ad(
    store, y, recipe: dict, *, asset_id, parent_run_id, scenario_id
) -> dict:
    """Prediction-based AD with CONFORMAL intervals: fit a forecaster + sktime ConformalIntervals
    on the history, predict a calibrated band over the recent window, flag points whose actual
    value falls OUTSIDE the band as anomalies. The forecaster is any catalog card (zero-shot TTM,
    classical, …); coverage = recipe.conformal.coverage (false-alarm = 1 − coverage)."""
    from ..substrate import resolver as R
    from sktime.forecasting.conformal import ConformalIntervals

    y = pd.Series(np.asarray(y, float)) if not isinstance(y, pd.Series) else y
    impute = recipe.get("impute") or (recipe.get("anomaly") or {}).get("impute")
    if impute:
        y = _impute(y, impute)
    spec = recipe.get("estimator")
    if not spec:
        raise ValueError("conformal AD needs an 'estimator' (a forecaster card)")
    card = spec
    if spec.get("model_id") and store is not None:
        card = model_store_get(store, spec["model_id"]) or spec
    merged = {
        **card,
        "params": {**(card.get("params") or {}), **(spec.get("params") or {})},
    }
    regime = R.training_regime(merged)
    coverage = float((recipe.get("conformal") or {}).get("coverage", 0.9))
    fh = recipe.get("fh") or list(
        range(1, max(2, len(y) // 5) + 1)
    )  # recent window to screen
    H = len(fh)
    y_train, y_test = y.iloc[:-H], y.iloc[-H:]

    ci = ConformalIntervals(R.resolve(merged))
    ci.fit(y_train, fh=fh)
    pi = ci.predict_interval(coverage=coverage)
    low = np.asarray(pi.iloc[:, 0])[:H]
    high = np.asarray(pi.iloc[:, 1])[:H]
    actual = np.asarray(y_test)[:H]
    out = (actual < low) | (actual > high)
    labels = np.zeros(len(y), dtype=int)
    labels[len(y) - H :][out] = 1
    indices = np.where(labels == 1)[0].tolist()
    n_anom = int(labels.sum())

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {
        "_id": run_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "asset_id": asset_id,
        "scenario_id": scenario_id,
        "task": "tsfm_anomaly_detection",
        "method": "conformal",
        "recipe": recipe,
        "training_regime": regime,
        "coverage": coverage,
        "n_anomalies": n_anom,
        "n_observations": len(y),
        "anomaly_indices": indices[:200],
        "created_at": _now(),
    }
    if store is not None:
        store.put(RUNS, rec)
    return {
        "run_id": run_id,
        "task": "tsfm_anomaly_detection",
        "method": "conformal",
        "n_anomalies": n_anom,
        "n_observations": len(y),
        "anomaly_indices_head": indices[:20],
        "labels": labels.tolist(),
        "training_regime": regime,
        "coverage": coverage,
        "parent_run_id": parent_run_id,
    }


def run_anomaly(
    store,
    y,
    recipe: dict,
    *,
    asset_id: str = "asset",
    parent_run_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """Anomaly run path. method='detector' (default): a detector card → fit → predict → dense
    labels (TSPulse zero-shot, SubLOF, PyOD). method='conformal': prediction-based AD — a
    forecaster + sktime ConformalIntervals → flag out-of-band points."""
    if recipe.get("method") == "conformal":
        return _conformal_ad(
            store,
            y,
            recipe,
            asset_id=asset_id,
            parent_run_id=parent_run_id,
            scenario_id=scenario_id,
        )
    from ..substrate import resolver as R

    y = pd.Series(np.asarray(y, float)) if not isinstance(y, pd.Series) else y
    spec = recipe.get("estimator") or recipe.get("detector")
    if not spec:
        raise ValueError("anomaly recipe needs an 'estimator' (a detector card)")
    card = spec
    if spec.get("model_id") and store is not None:
        card = model_store_get(store, spec["model_id"]) or spec
    merged = {
        **card,
        "params": {**(card.get("params") or {}), **(spec.get("params") or {})},
    }
    regime = R.training_regime(merged)
    block_audit = _validate_blocks(recipe)

    det = R.resolve(merged)
    n = len(y)
    impute = recipe.get("impute") or (recipe.get("anomaly") or {}).get("impute")
    if impute == "drop":
        y_fit = y.dropna()
        kept = np.where(y.notna().to_numpy())[
            0
        ]  # original positions retained after drop
    elif impute:
        y_fit = _impute(y, impute)  # interpolate | zero (length preserved)
        kept = np.arange(n)
    else:
        y_fit = y  # no imputation → let the model decide
        kept = np.arange(n)

    try:
        det.fit(y_fit)
        _, raw_idx = _anomaly_labels(det.predict(y_fit), len(y_fit))
    except ValueError as e:  # surface the model's own NaN error + hint
        if "nan" in str(e).lower() and not impute:
            raise ValueError(
                f"{type(det).__name__} received missing values and cannot handle NaN: {e} "
                "Set recipe['impute'] to 'interpolate', 'drop', or 'zero'."
            ) from e
        raise

    labels = np.zeros(n, dtype=int)  # map detector output back to ORIGINAL length
    orig = kept[np.asarray(raw_idx, dtype=int)] if raw_idx else np.array([], dtype=int)
    labels[orig] = 1
    indices = orig.tolist()
    n_anom = int(labels.sum())

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {
        "_id": run_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "asset_id": asset_id,
        "scenario_id": scenario_id,
        "task": "tsfm_anomaly_detection",
        "recipe": recipe,
        "training_regime": regime,
        "trained": regime != "zero_shot",
        "block_audit": block_audit,
        "n_anomalies": n_anom,
        "n_observations": len(y),
        "anomaly_indices": indices[:200],
        "created_at": _now(),
    }
    if store is not None:
        store.put(RUNS, rec)
    return {
        "run_id": run_id,
        "task": "tsfm_anomaly_detection",
        "n_anomalies": n_anom,
        "n_observations": len(y),
        "anomaly_indices_head": indices[:20],
        "labels": labels.tolist(),
        "training_regime": regime,
        "block_audit": block_audit,
        "parent_run_id": parent_run_id,
    }


def model_store_get(store, model_id):
    from ..stores import model_store

    return model_store.get_model(store, model_id)


def discover_components(store=None, task: str = "tsfm_forecasting") -> dict:
    """Everything the agent can mix-and-match for a task — including the TRAINING REGIME of each
    option, so the agent can reason cost vs. benefit and prefer zero-shot (no training) first.
    """
    from ..substrate import resolver as R

    fnd = R.foundation_forecasters()
    out = {
        "task": task,
        "scitype": R.TASK_TO_SCITYPE.get(task),
        "models_installed": R.discover(R.TASK_TO_SCITYPE.get(task, "forecaster"))[:50],
        "foundation_models": fnd,  # all zero-shot capable
        "zero_shot_models": fnd,  # default: predict with NO training
        "combiners": ["mean", "median", "min", "max", "weighted", "stack"],
        "metrics": ["smape", "mape", "mae", "mase"],
        "splitters": ["expanding", "sliding"],
        "training_regimes": {
            "zero_shot": "pretrained foundation model; fit() only loads weights + sets "
            "context, no training on your data — cheapest, the default path",
            "fit_on_series": "classical model (AutoARIMA/ETS/Theta/reduction) estimated on "
            "the series' own history — cheap, no separate training set",
            "fine_tune": "OPTIONAL escalation: adapt a foundation model on the data (pass "
            "fine-tune params or recipe.finetune) — most expensive, rarely needed",
        },
        "regime_hint": "try zero_shot first; escalate to fine_tune only if evaluate() warrants",
    }
    from ..reasoning import param_space as PS

    out["recipe_blocks"] = {  # run-time params the agent reasons
        "finetune": PS.FINETUNE_HINTS,
        "anomaly": PS.ANOMALY_HINTS,
    }
    from ..core import glossary as _glossary  # teach the vocabulary inline

    g = _glossary.glossary()
    out["glossary"] = g["terms"]
    out["workflow"] = g["workflow"]
    out["principles"] = g["principles"]
    if store is not None:
        from ..stores import model_store
        from ..stores import feature_store

        models = model_store.list_models(store, task_id=task)
        out["catalog_models"] = [
            {"model_id": m["model_id"], "training_regime": R.training_regime(m)}
            for m in models
        ]
        out["transforms"] = [
            f["feature_id"] for f in feature_store.find_features(store)
        ]
    return out
