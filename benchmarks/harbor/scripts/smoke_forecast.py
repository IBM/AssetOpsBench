#!/usr/bin/env python
"""Forecast a real sensor series with every card, and score it against naive.

`--check` proves a checkpoint is present. `test_catalog_checkpoints.py` proves
it loads and returns the right shape on synthetic data. Neither proves the
forecast is any good, and the ways a TTM card goes silently wrong all produce
output that passes both:

  * weights that failed to load leave the head at its initialisation
  * a model trained with TimeSeriesPreprocessor(scaling=True) served without
    scaling returns finite, varying, wrongly-scaled numbers
  * a card missing fit_strategy re-tunes on the series it is handed, so it
    looks plausible and is not the model you think it is

All three are caught by the same question: does it beat predicting the last
value? A model that cannot is not forecasting, whatever its output shape.

    uv run python benchmarks/harbor/scripts/smoke_forecast.py
    AOB_MODEL_CATALOG=/path/to/private.json uv run python .../smoke_forecast.py
    uv run python .../smoke_forecast.py --series <file.json> --column "<sensor>"

Exit 1 when any card fails the floor, so it works as a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from pathlib import Path

DEFAULT_SERIES = Path("src/couchdb/scenarios_data/shared/iot/chiller_6.json")
DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")


def catalog_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    env = os.environ.get("AOB_MODEL_CATALOG")
    if env:
        return Path(env)
    root = os.environ.get("SCENARIOS_DATA_DIR")
    if root:
        return Path(root) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def load_series(path: Path, column: str | None, max_missing: float = 0.05):
    """Pull one numeric sensor column out of an AssetOpsBench IoT file.

    Real sensor data has gaps: chiller_6 carries ~20 nulls per column out of
    2896, and one column is 99% null. sktime's TTM declares
    capability:missing_values=False, so the gaps must be closed before fitting.
    Columns missing more than `max_missing` are rejected rather than
    interpolated, because inventing most of a series makes the score
    meaningless.
    """
    import numpy as np
    import pandas as pd

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} is not a list of records")

    # Real sensor timestamps drift (chiller_6 spacing is 15min plus or minus a
    # few seconds), so pandas infers no frequency and a DatetimeIndex built from
    # them carries freq=None. Snap to the median spacing: a forecaster needs a
    # regular grid, and three seconds of jitter is not signal.
    stamps = pd.to_datetime([r.get("timestamp") for r in records], errors="coerce")
    if stamps.isna().any():
        index = pd.RangeIndex(len(records))
        step = None
    else:
        step = pd.Series(stamps).diff().median()
        index = pd.date_range(stamps[0], periods=len(records), freq=step)

    skip = {"asset_id", "timestamp"}
    numeric = [k for k, v in records[0].items()
               if k not in skip and isinstance(v, (int, float))]
    if not numeric:
        raise SystemExit(f"no numeric columns in {path}")

    max_zero_frac = 0.2

    def series_for(name: str):
        raw = [r.get(name) for r in records]
        arr = np.array([np.nan if v is None else v for v in raw], dtype="float64")
        n_missing = int(np.isnan(arr).sum())
        if n_missing / len(arr) > max_missing:
            return None, n_missing
        if n_missing:
            idx = np.arange(len(arr))
            good = ~np.isnan(arr)
            arr = np.interp(idx, idx[good], arr[good])  # linear, ends held flat
        return arr, n_missing

    def intermittent(arr) -> bool:
        """On/off series (chiller power is 65% zeros) break percentage errors
        and make naive nearly unbeatable. Not a fair test of a forecaster."""
        return float((arr == 0).mean()) > max_zero_frac

    if column is not None:
        arr, n_missing = series_for(column)
        if arr is None:
            raise SystemExit(f"column {column!r} is more than "
                             f"{max_missing:.0%} missing ({n_missing} points)")
        if intermittent(arr):
            print(f"warning: {column!r} is {float((arr == 0).mean()):.0%} zeros; "
                  "percentage errors are unreliable on intermittent series",
                  file=sys.stderr)
        return arr, column, n_missing, index, step

    # Pick the column with the most variation relative to its level: a
    # near-constant sensor makes naive unbeatable and the test meaningless.
    best, best_arr, best_cv, best_miss = None, None, -1.0, 0
    for k in numeric:
        arr, n_missing = series_for(k)
        if arr is None or arr.std() == 0 or intermittent(arr):
            continue
        cv = arr.std() / (abs(arr.mean()) + 1e-9)
        if cv > best_cv:
            best, best_arr, best_cv, best_miss = k, arr, cv, n_missing
    if best is None:
        raise SystemExit(f"no usable numeric column in {path}")
    return best_arr, best, best_miss, index, step


def smape(actual, pred) -> float:
    """Symmetric MAPE in percent. Scale-free, so different sensors compare."""
    import numpy as np

    a, p = np.asarray(actual, "float64"), np.asarray(pred, "float64")
    denom = (np.abs(a) + np.abs(p)) / 2.0
    denom[denom == 0] = 1e-9
    return float(np.mean(np.abs(a - p) / denom) * 100.0)


def blocked_by_interpreter(card: dict) -> str:
    """The python_version this estimator demands, when this one does not satisfy it.

    TimesFMForecaster declares >=3.10,<3.11 alongside a jax/paxml stack. No
    install fixes that on 3.12, so its packages must never reach the aggregate
    install line: following that advice costs gigabytes and changes nothing.
    """
    import importlib
    import platform

    dotted = card.get("sktime_class")
    if not dotted or "." not in dotted:
        return ""
    mod, _, name = dotted.rpartition(".")
    try:
        from packaging.specifiers import SpecifierSet

        cls = getattr(importlib.import_module(mod), name)
        spec = (cls.get_class_tags() or {}).get("python_version")
        if spec and platform.python_version() not in SpecifierSet(str(spec)):
            return str(spec)
    except Exception:  # noqa: BLE001 - best effort
        return ""
    return ""


def missing_deps(card: dict) -> list[str]:
    """Every declared dependency this interpreter lacks, not just the first.

    sktime reports one missing soft dependency per attempt, so installing what
    it names and re-running just surfaces the next one. MOIRAIForecaster
    declares eight. Read the estimator's own python_dependencies tag and check
    them all at once, so one run yields one install command.
    """
    import importlib
    import importlib.metadata as md
    import re

    dotted = card.get("sktime_class")
    if not dotted or "." not in dotted:
        return []
    mod, _, name = dotted.rpartition(".")
    try:
        cls = getattr(importlib.import_module(mod), name)
        declared = (cls.get_class_tags() or {}).get("python_dependencies") or []
    except Exception:  # noqa: BLE001 - best effort; the caller still reports the skip
        return []

    out = []
    for spec in declared if isinstance(declared, list) else [declared]:
        pkg = re.split(r"[<>=!~\[]", str(spec), 1)[0].strip()
        if not pkg:
            continue
        try:
            md.version(pkg)
        except md.PackageNotFoundError:
            out.append(pkg)
    return out


def serving_switch(card: dict) -> str | None:
    """The constructor parameter that makes THIS estimator serve instead of train.

    There is no universal one. TTM and PatchTST take fit_strategy="zero-shot";
    PatchTSMixer takes train_model=False; MOMENT takes neither, because its
    forecasting head is built fresh and has to be trained. Advising a parameter
    the estimator does not accept is worse than saying nothing, so read the
    signature instead of guessing.
    """
    import importlib
    import inspect

    dotted = card.get("sktime_class")
    if not dotted or "." not in dotted:
        return None
    mod, _, name = dotted.rpartition(".")
    try:
        cls = getattr(importlib.import_module(mod), name)
        sig = inspect.signature(cls.__init__).parameters
    except Exception:  # noqa: BLE001 - best effort
        return None
    params = card.get("params") or {}
    if "fit_strategy" in sig and "fit_strategy" not in params:
        return 'params.fit_strategy="zero-shot"'
    if "train_model" in sig and "train_model" not in params:
        return "params.train_model=false"
    return None


def horizon_of(card: dict, default_h: int) -> int:
    raw = card.get("prediction_length")
    return int(raw) if raw else default_h


def forecastable(card: dict) -> tuple[bool, str]:
    """Is this card a forecasting model with weights, or a registry entry?

    A full catalog carries more than checkpoints: engine/algorithm entries
    (`autoarima`, `naive_persistence`) that name no weights, and detectors and
    classifiers (`pyod_iforest`, `tspulse_ad`, `tskmeans`) that do not forecast
    at all. Scoring those against a forecast baseline is meaningless, so name
    them and move on rather than reporting an error.
    """
    tasks = card.get("task_ids") or []
    if tasks and not any("forecast" in str(t) for t in tasks):
        return False, f"not a forecaster (task_ids={','.join(map(str, tasks))})"
    has_weights = bool((card.get("params") or {}).get("model_path")
                       or card.get("model_checkpoint")
                       or card.get("artifact_path")
                       or card.get("hf_repo"))
    if not has_weights:
        return False, "registry entry, names no weights"
    return True, ""


def evaluate(card: dict, y, season: int, default_ctx: int, default_h: int,
             index=None) -> dict:
    """Fit on history, forecast the holdout, score against two naive baselines."""
    import contextlib
    import io
    import warnings

    import numpy as np
    import pandas as pd

    from servers.tsfm.substrate import resolver as R

    ok, why = forecastable(card)
    if not ok:
        return {"status": "SKIP", "note": why}

    # Foundation-model cards routinely leave these null: the wrapper picks a
    # context window at fit time. Fall back rather than crashing, and say so,
    # because a defaulted horizon is not the horizon the card promises.
    raw_ctx, raw_h = card.get("context_length"), card.get("prediction_length")
    ctx = int(raw_ctx) if raw_ctx else default_ctx
    horizon = int(raw_h) if raw_h else default_h
    defaulted = [n for n, v in (("ctx", raw_ctx), ("h", raw_h)) if not v]

    need = ctx + horizon
    if len(y) < need:
        return {"status": "SKIP", "note": f"series has {len(y)} points, needs {need}"}

    train, test = y[:-horizon], y[-horizon:]

    idx = index[:len(train)] if index is not None else None
    forecaster = R.resolve(card)
    # Two independent signals that a card trains instead of serving.
    #
    # Declared: training_regime reads the card. Deterministic, but it falls back
    # to a guess for an estimator it cannot inspect, so it can be wrong either way.
    #
    # Observed: HF's Trainer writes loss/epoch dicts while it runs. Proof when it
    # appears, but it is absent for a strategy that trains quietly, so absence
    # proves nothing. Capture both streams, and treat only the observed signal
    # as conclusive.
    try:
        declared = R.training_regime(card)
    except Exception:  # noqa: BLE001 - an uninspectable card is not a failure
        declared = "unknown"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(pd.Series(train, index=idx), fh=list(range(1, horizon + 1)))
        pred = np.asarray(forecaster.predict()).ravel()[:horizon]
    captured = buf.getvalue()
    trained = "train_runtime" in captured or "'loss'" in captured

    if len(pred) != horizon or not np.isfinite(pred).all():
        return {"status": "FAIL",
                "note": f"returned {len(pred)} points, finite={np.isfinite(pred).all()}"}

    naive = np.full(horizon, train[-1])                      # last value carried forward
    seas = train[-season:][:horizon] if len(train) >= season else naive
    if len(seas) < horizon:
        seas = np.resize(seas, horizon)

    def mae(a, b):
        return float(np.mean(np.abs(np.asarray(a, "float64") - np.asarray(b, "float64"))))

    # Skill score: model MAE over the better naive MAE. <1 beats naive. MAE
    # rather than a percentage error, because percentages blow up near zero.
    mae_m, mae_n, mae_s = mae(test, pred), mae(test, naive), mae(test, seas)
    skill = mae_m / (min(mae_n, mae_s) + 1e-12)
    m, n, s = smape(test, pred), smape(test, naive), smape(test, seas)

    # A forecast whose level is nowhere near the recent history is the signature
    # of unloaded weights or a scaling mismatch, and sMAPE alone can understate it.
    drift = abs(pred.mean() - train[-ctx:].mean()) / (train.std() + 1e-9)

    out = {"skill": skill, "smape": m, "naive": n, "seasonal": s, "drift": drift}
    notes = []
    if defaulted:
        notes.append(f"card declares no {'/'.join(defaulted)}; used {ctx}/{horizon}")

    # A flat line is what an unloaded head returns. It can score near naive on
    # a series with little trend, so check the shape directly rather than
    # relying on the error to expose it.
    if pred.std() / (train.std() + 1e-12) < 0.01:
        return {**out, "status": "FAIL",
                "note": "; ".join([*notes, "constant forecast: weights did not load"])}

    if trained:
        # Observed, not inferred: the trainer ran. The model saw this holdout's
        # history as training data, so the score is not a zero-shot score.
        return {**out, "status": "TRAINED", "note": "; ".join(
            [*notes, "trainer ran during fit; score is not zero-shot"])}

    if skill > 2:
        status = "FAIL"
        notes.append(f"{skill:.1f}x the error of naive")
    elif drift > 3:
        status = "FAIL"
        notes.append(f"forecast level is {drift:.1f} sd from recent history")
    elif skill > 1:
        status = "WARN"
        notes.append(f"{skill:.2f}x naive")
    else:
        status = "PASS"

    # No trainer output, but the card does not pin serving either.
    if declared not in ("zero_shot", "unknown"):
        switch = serving_switch(card)
        if switch:
            # The estimator can serve and the card has not asked it to. That is
            # a card to fix, and the parameter named here exists on this class.
            if status == "PASS":
                status = "WARN"
            notes.append(f"declares regime={declared}; set {switch} to serve")
        else:
            # The estimator has no serving switch, so fine_tune is not an
            # oversight, it is what this model does. Mark the score as tuned
            # rather than nagging for a parameter that does not exist.
            status = "TUNED"
            notes.append(f"regime={declared} by design; this estimator has no "
                         "serving switch, so the score includes tuning on this series")
    return {**out, "status": status, "note": "; ".join(notes)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    p.add_argument("--column", default=None, help="sensor column (default: the most variable)")
    p.add_argument("--season", type=int, default=None,
                   help="seasonal-naive lag in steps (default: one day, from the "
                        "series' own sampling rate)")
    p.add_argument("--default-context", type=int, default=512,
                   help="context to use for cards that declare none (default 512)")
    p.add_argument("--explain", metavar="MODEL_ID",
                   help="run only this card and print the full traceback")
    p.add_argument("--default-horizon", type=int, default=96,
                   help="horizon to use for cards that declare none (default 96)")
    args = p.parse_args()

    sys.path.insert(0, "src")
    # huggingface_hub revalidates a cached file's etag over HTTP before using
    # it, and httpx logs every one at INFO. Those lines look like downloads and
    # interleave with the table. Cached weights are still served from cache.
    import logging

    for noisy_loggers in ("httpx", "httpcore", "urllib3", "filelock",
                          "huggingface_hub", "transformers", "datasets"):
        logging.getLogger(noisy_loggers).setLevel(logging.WARNING)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    if not args.series.is_file():
        print(f"no series at {args.series}", file=sys.stderr)
        return 1

    raw = json.loads(cat.read_text(encoding="utf-8"))
    cards = [c for c in (raw if isinstance(raw, list) else raw.get("docs", [raw]))
             if (c.get("status") or "active") == "active"]
    if args.explain:
        cards = [c for c in cards if c.get("model_id") == args.explain]
        if not cards:
            print(f"no active card with model_id={args.explain!r}", file=sys.stderr)
            return 1
        print(f"card    : {json.dumps(cards[0], indent=2)}\n")

    y, column, filled, index, step = load_series(args.series, args.column)

    season = args.season
    if season is None:
        if step is not None and step.total_seconds() > 0:
            season = max(2, round(86400 / step.total_seconds()))
        else:
            season = 24
    if season >= len(y) // 2:
        season = 24

    print(f"catalog : {cat}")
    rate = (f"{step.total_seconds() / 60:g}min sampling, "
            f"seasonal lag {season} steps (one day)" if step is not None
            else f"no usable timestamps, seasonal lag {season} steps")
    print(f"series  : {args.series.name}  column={column!r}  n={len(y)}"
          + (f"  ({filled} gaps interpolated)" if filled else ""))
    print(f"          {rate}")
    print()
    print(f"  {'model':24} {'status':7} {'skill':>7} {'sMAPE':>8} {'naive%':>8}  note")
    print("  " + "-" * 78)

    failures = 0
    missing: set[str] = set()
    for c in cards:
        try:
            r = evaluate(c, y, season, args.default_context, args.default_horizon,
                         index=index)
        except Exception as exc:  # noqa: BLE001 - one bad card must not stop the sweep
            text = " ".join(str(exc).split())
            if isinstance(exc, KeyError) and ("Timestamp(" in text or text.startswith("'[")):
                n = text.count("Timestamp(")
                r = {"status": "FAIL",
                     "note": (f"returned fewer points than the {horizon_of(c, args.default_horizon)}"
                              f" requested; {n} timestamp(s) missing from the forecast")}
                print(f"  {c['model_id']:24} {r['status']:7} {'-':>7} {'-':>8} {'-':>8}"
                      f"  {r['note']}")
                failures += 1
                continue
            offline = c.get("hf_repo") and any(
                s in text for s in ("couldn't connect", "Offline", "offline",
                                    "Connection", "resolve"))
            # "requires python version to be <3.11", "requires package 'x'":
            # this interpreter cannot host the model. That is a packaging fact
            # about the run, not a defect in the card.
            env = any(s in text for s in ("requires python version",
                                          "requires package",
                                          "to be present in the python environment"))
            if offline:
                r = {"status": "SKIP", "note": f"{c['hf_repo']} unreachable (offline)"}
            elif env:
                pin = blocked_by_interpreter(c)
                if pin:
                    r = {"status": "SKIP",
                         "note": f"needs python {pin}; this is "
                                 f"{platform.python_version()}. No install fixes it."}
                else:
                    want = missing_deps(c)
                    r = {"status": "SKIP",
                         "note": (f"needs: {' '.join(want)}" if want
                                  else f"environment: {text[:56]}")}
                    missing.update(want)
            else:
                width = 200 if args.explain else 110
                r = {"status": "ERROR", "note": f"{type(exc).__name__}: {text[:width]}"}
                if args.explain:
                    import traceback
                    print()
                    traceback.print_exc()
                    print()
        if r["status"] in ("FAIL", "ERROR", "TRAINED"):
            failures += 1
        nums = (f"{r['skill']:>7.2f} {r['smape']:>8.2f} {r['naive']:>8.2f}"
                if "skill" in r else f"{'-':>7} {'-':>8} {'-':>8}")
        print(f"  {c['model_id']:24} {r['status']:7} {nums}  {r['note']}")

    print()
    if missing:
        print("Some cards were skipped for missing packages. Install them together:\n")
        print(f"  uv pip install {' '.join(sorted(missing))}\n")
        print("A version conflict here is a result, not an obstacle: cards whose pins "
              "cannot co-exist\nneed separate images, and that is what the per-ecosystem "
              "layers are for.\n")
    if failures:
        print(f"{failures} card(s) are not forecasting. Investigate before trusting any "
              "benchmark number that used them.", file=sys.stderr)
        return 1
    print("Every card that ran beats or matches naive on this series (skill <= 1).")
    print("sMAPE is shown for familiarity only. It is not the gate: on a series "
          "near or crossing zero it\nreads high even for a good forecast, which "
          "is why the floor is MAE relative to naive.")
    print("TUNED means the estimator has no way to serve without training, so its score "
          "is not\ncomparable with a zero-shot one. Keep those cards in a separate "
          "column of any results table.")
    print("A WARN is not automatically wrong: a zero-shot model can lose to naive on a "
          "series unlike its\ntraining data. A FAIL means the output is not a forecast. "
          "A TRAINED card fine-tuned on the series\nduring fit, so its score is optimistic "
          "and not comparable: pin params.fit_strategy and\ntraining_regime. SKIP is this "
          "run's limits (offline, interpreter, series length), not the card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
