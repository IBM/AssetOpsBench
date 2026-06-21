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


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _metric(name):
    from sktime.performance_metrics.forecasting import (
        MeanAbsolutePercentageError, MeanAbsoluteError, MeanAbsoluteScaledError)
    return {"smape": MeanAbsolutePercentageError(symmetric=True),
            "mape": MeanAbsolutePercentageError(),
            "mae": MeanAbsoluteError(),
            "mase": MeanAbsoluteScaledError()}.get(name, MeanAbsolutePercentageError(symmetric=True))


def _resolve_estimator(spec: dict, store=None):
    """estimator spec → (name, sktime_estimator). spec: {name?, model_id?|sktime_class, params?}."""
    from tsfm.substrate import resolver as R
    name = spec.get("name") or spec.get("model_id") or spec.get("sktime_class", "est").split(".")[-1]
    if spec.get("model_id") and store is not None:
        from tsfm.stores import model_store
        card = model_store.get_model(store, spec["model_id"])
        if not card:
            raise ValueError(f"model '{spec['model_id']}' not in catalog")
        return name, R.resolve(card)
    return name, R.resolve(spec)            # sktime_class + params directly


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
        else:                                # mean | median | min | max
            base = EnsembleForecaster(members, aggfunc=combine)
    else:
        _, base = _resolve_estimator(recipe["estimator"], store)

    # 2) optional transforms with automatic inverse (AutoAI-TS reverse-order, via sktime)
    transforms = recipe.get("transforms") or []
    if transforms:
        from sktime.forecasting.compose import TransformedTargetForecaster
        from tsfm.substrate import resolver as R
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
    forecaster.fit(y_train, fh=fh)                       # fit = load weights + set context
    y_pred = np.asarray(forecaster.predict())[:h]
    yt = np.asarray(y_test)[:len(y_pred)]
    try:
        score = float(metric(yt, y_pred, y_train=np.asarray(y_train)))  # MASE etc.
    except TypeError:
        score = float(metric(yt, y_pred))
    return score, metric.__class__.__name__, 1


def _recipe_regime(recipe, store) -> str:
    """Cheapest regime that covers the whole recipe: zero_shot only if every estimator is
    pretrained-zero-shot and no fine-tune is requested; else fit_on_series (or fine_tune)."""
    from tsfm.substrate import resolver as R
    if recipe.get("finetune"):
        return "fine_tune"
    specs = recipe["ensemble"]["members"] if "ensemble" in recipe else [recipe["estimator"]]
    regimes = set()
    for s in specs:
        card = s
        if s.get("model_id") and store is not None:
            from tsfm.stores import model_store
            card = model_store.get_model(store, s["model_id"]) or s
        merged = {**card, "params": {**(card.get("params") or {}), **(s.get("params") or {})}}
        regimes.add(R.training_regime(merged))
    if "fine_tune" in regimes:
        return "fine_tune"
    return "zero_shot" if regimes == {"zero_shot"} else "fit_on_series"


def run_recipe(store, y, recipe: dict, *, asset_id: str = "asset",
               parent_run_id: Optional[str] = None, scenario_id: Optional[str] = None) -> dict:
    """Compile → backtest → final forecast → per-member diagnostics → persist run (with lineage)."""
    y = pd.Series(np.asarray(y, float)) if not isinstance(y, pd.Series) else y
    fc = build_forecaster(recipe, store)
    regime = _recipe_regime(recipe, store)
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
        cov = recipe["conformal"].get("coverage", 0.9) if isinstance(recipe["conformal"], dict) else 0.9
        try:
            intervals = fc.predict_interval(coverage=cov).round(4).head(3).to_dict()
        except Exception as e:
            intervals = {"error": str(e)[:80]}

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {"_id": run_id, "run_id": run_id, "parent_run_id": parent_run_id,
           "asset_id": asset_id, "scenario_id": scenario_id, "recipe": recipe,
           "training_regime": regime, "trained": regime != "zero_shot",
           "metric": metric_name, "backtest_score": round(score, 4), "folds": folds,
           "per_member_score": per_member, "forecast_head": forecast[:5],
           "prediction_interval": intervals, "created_at": _now()}
    if store is not None:
        store.put(RUNS, rec)
    return {"run_id": run_id, "backtest_score": round(score, 4), "metric": metric_name,
            "training_regime": regime, "trained": regime != "zero_shot",
            "per_member_score": per_member, "forecast_head": forecast[:5],
            "prediction_interval": intervals, "parent_run_id": parent_run_id,
            "improved": (parent_run_id is not None and store is not None
                         and score < (store.get(RUNS, parent_run_id) or {}).get("backtest_score", 9e9))}


TABULAR_TASKS = {"tsfm_regression", "tsfm_classification", "tsfm_clustering"}


def _as_panel(X):
    """Accept (n, T) univariate or (n, C, T) multivariate → (n, C, T)."""
    X = np.asarray(X, float)
    return X[:, None, :] if X.ndim == 2 else X


def _lib_features(X, subset=None):
    """Dependency-free 'FeatureUnion': apply the FLOps extractor library per instance/channel."""
    from tsfm.reasoning import feature_selection as FS
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
        return _lib_features(X)                              # default = full library
    parts, names = [], []
    for t in transforms:
        if t.get("flops_select"):
            from tsfm.reasoning import feature_selection as FS
            F, nm = _lib_features(X)
            if y is not None:
                keep = _flops_columns(F, np.asarray(y), nm, t.get("flops_select"))
                idx = [nm.index(k) for k in keep]
                F, nm = F[:, idx], keep
            parts.append(F); names += nm
        elif t.get("extractors"):
            F, nm = _lib_features(X, t["extractors"]); parts.append(F); names += nm
        elif t.get("sktime_class"):
            from tsfm.substrate import resolver as R
            tr = R.resolve(t)
            Ft = np.nan_to_num(np.asarray(tr.fit_transform(_as_panel(X))).reshape(len(X), -1))
            parts.append(Ft)
            names += [f"{t['sktime_class'].split('.')[-1]}_{k}" for k in range(Ft.shape[1])]
    F = np.column_stack(parts) if parts else _lib_features(X)[0]
    return np.nan_to_num(F), names


def _flops_columns(F, y, names, cfg):
    """Reuse the v2 multi-config scorers to select tabular feature COLUMNS by relevance to y."""
    from tsfm.reasoning import feature_selection as FS
    scorers = (cfg.get("scorers") if isinstance(cfg, dict) else None) or ["corr", "f_test", "mutual_info", "model"]
    use = [s for s in scorers if s in FS._SCORERS] or ["corr"]
    ranks = np.zeros(len(names))
    for s in use:
        sv = np.asarray(FS._SCORERS[s](F, y, names))
        order = (-sv).argsort(); rk = np.empty(len(names)); rk[order] = np.arange(1, len(names) + 1)
        ranks += rk
    mean_rank = ranks / len(use)
    cut = (cfg.get("top_k") if isinstance(cfg, dict) else None) or max(1, len(names) // 2)
    keep_idx = np.argsort(mean_rank)[:cut]
    return [names[i] for i in sorted(keep_idx)]


def run_tabular_recipe(store, X, recipe: dict, *, y=None, asset_id: str = "asset",
                       parent_run_id: Optional[str] = None, scenario_id: Optional[str] = None) -> dict:
    """Series→tabular run path: FeatureUnion(extractors) → estimator, for regression /
    classification / clustering. CV-scored (supervised) or silhouette (clustering). Same recipe
    grammar + persistence + lineage as forecasting; the agent mixes-and-matches features here too."""
    from tsfm.substrate import resolver as R
    from sklearn.model_selection import cross_val_score
    task = recipe.get("task", "tsfm_classification")
    F, feat_names = _tabular_features(X, recipe, y)
    regime = _recipe_regime(recipe, store) if "estimator" in recipe or "ensemble" in recipe else "fit_on_series"
    _, est = _resolve_estimator(recipe["estimator"], store)

    metric, score = None, None
    if task == "tsfm_clustering":
        from sklearn.metrics import silhouette_score
        labels = est.fit_predict(F) if hasattr(est, "fit_predict") else est.fit(F).predict(F)
        try:
            score, metric = round(float(silhouette_score(F, labels)), 4), "silhouette"
        except Exception:
            score, metric = None, "silhouette"
        preds = np.asarray(labels)[:10].tolist()
    else:
        y = np.asarray(y)
        scoring = "accuracy" if task == "tsfm_classification" else "r2"
        nfold = int(min(5, max(2, np.min(np.bincount(y)) if task == "tsfm_classification" else len(y) // 3)))
        try:
            score = round(float(cross_val_score(est, F, y, cv=nfold, scoring=scoring).mean()), 4)
        except Exception as e:
            score = f"err:{type(e).__name__}"
        metric = scoring
        est.fit(F, y); preds = np.asarray(est.predict(F))[:10].tolist()

    run_id = f"run:{uuid.uuid4().hex[:10]}"
    rec = {"_id": run_id, "run_id": run_id, "parent_run_id": parent_run_id, "task": task,
           "asset_id": asset_id, "scenario_id": scenario_id, "recipe": recipe,
           "training_regime": regime, "n_features": F.shape[1], "feature_names": feat_names[:40],
           "metric": metric, "cv_score": score, "predictions_head": preds, "created_at": _now()}
    if store is not None:
        store.put(RUNS, rec)
    return {"run_id": run_id, "task": task, "metric": metric, "cv_score": score,
            "n_features": F.shape[1], "feature_names": feat_names[:40],
            "predictions_head": preds, "training_regime": regime, "parent_run_id": parent_run_id}


def discover_components(store=None, task: str = "tsfm_forecasting") -> dict:
    """Everything the agent can mix-and-match for a task — including the TRAINING REGIME of each
    option, so the agent can reason cost vs. benefit and prefer zero-shot (no training) first."""
    from tsfm.substrate import resolver as R
    fnd = R.foundation_forecasters()
    out = {"task": task, "scitype": R.TASK_TO_SCITYPE.get(task),
           "models_installed": R.discover(R.TASK_TO_SCITYPE.get(task, "forecaster"))[:50],
           "foundation_models": fnd,                       # all zero-shot capable
           "zero_shot_models": fnd,                        # default: predict with NO training
           "combiners": ["mean", "median", "min", "max", "weighted", "stack"],
           "metrics": ["smape", "mape", "mae", "mase"],
           "splitters": ["expanding", "sliding"],
           "training_regimes": {
               "zero_shot": "pretrained foundation model; fit() only loads weights + sets "
                            "context, no training on your data — cheapest, the default path",
               "fit_on_series": "classical model (AutoARIMA/ETS/Theta/reduction) estimated on "
                                "the series' own history — cheap, no separate training set",
               "fine_tune": "OPTIONAL escalation: adapt a foundation model on the data (pass "
                            "fine-tune params or recipe.finetune) — most expensive, rarely needed"},
           "regime_hint": "try zero_shot first; escalate to fine_tune only if evaluate() warrants"}
    if store is not None:
        from tsfm.stores import model_store
        from tsfm.stores import feature_store
        models = model_store.list_models(store, task_id=task)
        out["catalog_models"] = [
            {"model_id": m["model_id"], "training_regime": R.training_regime(m)} for m in models]
        out["transforms"] = [f["feature_id"] for f in feature_store.find_features(store)]
    return out
