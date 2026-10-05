#!/usr/bin/env python
"""Prove the non-forecasting cards actually work, the way smoke_forecast does.

`preload_models.py --check` proves a checkpoint is present. Schema validation
proves a card is well formed. `smoke_forecast.py` covers the forecasters and
skips everything else, so the detectors, the classifier and the clusterer in
the catalog have never been run by anything.

Each task gets a problem with a known answer, built from a real sensor series,
plus the baseline a useless model would score:

  anomaly detection   spikes injected at known indices; score recall within a
                      tolerance window against a random detector flagging the
                      same number of points
  classification      two classes separated by a level shift; score accuracy
                      against always predicting the majority class
  clustering          three groups (baseline, shifted, damped); score adjusted
                      Rand index, which is 0 for a random assignment

A model that loaded but is not working fails the floor. A model that never
loaded raises, and is reported as an error rather than a bad score.

    uv run python benchmarks/harbor/scripts/smoke_tasks.py
    AOB_MODEL_CATALOG=/path/to/private.json uv run python .../smoke_tasks.py
    uv run python .../smoke_tasks.py --explain tspulse_ad

Exit 1 when any card fails, so it works as a gate beside smoke_forecast.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SERIES = Path("src/couchdb/scenarios_data/shared/iot/chiller_6.json")
DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")

N_ANOMALIES = 12
TOLERANCE = 5           # a detection this many steps from an injection counts
WINDOW = 64             # panel window length for classification and clustering
PER_GROUP = 20


def catalog_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    if os.environ.get("AOB_MODEL_CATALOG"):
        return Path(os.environ["AOB_MODEL_CATALOG"])
    if os.environ.get("SCENARIOS_DATA_DIR"):
        return Path(os.environ["SCENARIOS_DATA_DIR"]) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def task_of(card: dict) -> str | None:
    for t in card.get("task_ids") or []:
        t = str(t)
        if "anomaly" in t:
            return "anomaly_detection"
        if "classification" in t:
            return "classification"
        if "clustering" in t:
            return "clustering"
    return None


def panel(y, specs):
    """Windows of the real series, transformed per group. Returns numpy3D + labels."""
    import numpy as np

    X, labels = [], []
    for g, (scale, shift) in enumerate(specs):
        for i in range(PER_GROUP):
            seg = y[i * WINDOW:(i + 1) * WINDOW]
            if len(seg) < WINDOW:
                break
            X.append(seg * scale + shift)
            labels.append(g)
    return np.asarray(X)[:, None, :], np.asarray(labels)


def eval_anomaly(card, y, index, column) -> dict:
    import numpy as np
    import pandas as pd

    from servers.tsfm.substrate import resolver as R

    rng = np.random.default_rng(0)
    arr = y.copy()
    sd = float(arr.std())
    spots = np.sort(rng.choice(np.arange(200, len(arr) - 200),
                               size=N_ANOMALIES, replace=False))
    for s in spots:
        arr[s:s + 3] += sd * 6 * (1 if rng.random() > 0.5 else -1)
    X = pd.DataFrame({column: arr}, index=index)

    det = R.resolve(card)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = det.fit_predict(X)

    flagged = np.asarray(out["ilocs"] if hasattr(out, "columns") and "ilocs" in out
                         else np.asarray(out)).ravel()
    if flagged.size == 0:
        return {"status": "FAIL", "note": "flagged nothing; a detector that never "
                                          "fires cannot be evaluated"}
    if flagged.size > len(arr) * 0.25:
        return {"status": "FAIL", "score": 1.0, "baseline": 1.0,
                "note": f"flagged {flagged.size}/{len(arr)} points; that is not detection"}

    def recall_at(tol: int) -> float:
        return sum(any(abs(f - s) <= tol for f in flagged) for s in spots) / len(spots)

    sweep = {tol: recall_at(tol) for tol in (TOLERANCE, 16, 32, 64, 128)}
    hits = round(sweep[TOLERANCE] * len(spots))
    recall = sweep[TOLERANCE]
    # A random detector flagging the same count: chance it lands in at least one
    # of an injection's 2*TOLERANCE+1 tolerant slots.
    p = flagged.size / len(arr)
    baseline = 1 - (1 - p) ** (2 * TOLERANCE + 1)
    detail = f"{hits}/{len(spots)} injected spikes found, {flagged.size} points flagged"

    # Where does recall saturate? If a wider tolerance finds the spikes, the
    # detector is working and reporting at window resolution, which is a
    # different finding from not detecting at all.
    # Only interesting when the tight tolerance MISSED. Take the smallest
    # tolerance that finds them, which is roughly the detector's resolution.
    coarse = None
    if sweep[TOLERANCE] < 0.5:
        coarse = min((tol for tol, r in sorted(sweep.items()) if r >= 0.5),
                     default=None)
    if coarse:
        detail += (f"; at tolerance {coarse} recall is {sweep[coarse]:.2f}, so it "
                   "detects at window resolution rather than per sample")

    if recall <= baseline:
        verdict = "FAIL" if not coarse else "WARN"
        return {"status": verdict, "score": recall, "baseline": baseline,
                "note": (f"no better than random at tolerance {TOLERANCE}; {detail}")}
    if recall < 0.5:
        return {"status": "WARN", "score": recall, "baseline": baseline, "note": detail}
    return {"status": "PASS", "score": recall, "baseline": baseline, "note": detail}


def eval_classification(card, y) -> dict:
    import numpy as np

    from servers.tsfm.substrate import resolver as R

    X, labels = panel(y, [(1.0, 0.0), (1.0, 3 * float(y.std()))])
    idx = np.arange(len(labels))
    np.random.default_rng(0).shuffle(idx)
    cut = int(len(idx) * 0.6)
    tr, te = idx[:cut], idx[cut:]

    clf = R.resolve(card)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X[tr], labels[tr])
        pred = np.asarray(clf.predict(X[te])).ravel()

    acc = float(np.mean(pred == labels[te]))
    baseline = max(float(np.mean(labels[te] == c)) for c in set(labels.tolist()))
    detail = f"{len(tr)} train / {len(te)} test windows, two classes"
    if acc <= baseline:
        return {"status": "FAIL", "score": acc, "baseline": baseline,
                "note": f"no better than predicting the majority class; {detail}"}
    return {"status": "PASS", "score": acc, "baseline": baseline, "note": detail}


def shape_panel():
    """Groups that differ by WAVEFORM at matched level and amplitude.

    The level/amplitude problem is unfair to shape-based clusterers, which
    z-normalise each window by design: KernelKMeans scores 0.03 there and 1.00
    here. Scoring the better of the two problems asks "can this estimator
    separate groups at all" rather than "does it use the cue I happened to
    pick".
    """
    import numpy as np

    t = np.arange(WINDOW)
    rng = np.random.default_rng(0)
    waves = [np.sin(2 * np.pi * t / 32),                 # smooth
             np.sign(np.sin(2 * np.pi * t / 32)),        # square
             (t % 32) / 16 - 1]                          # sawtooth
    X, labels = [], []
    for g, wave in enumerate(waves):
        for _ in range(PER_GROUP):
            X.append(wave + rng.normal(0, 0.05, WINDOW))
            labels.append(g)
    return np.asarray(X)[:, None, :], np.asarray(labels)


def eval_clustering(card, y) -> dict:
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from servers.tsfm.substrate import resolver as R

    sd = float(y.std())
    problems = {
        "level/amplitude": panel(y, [(1.0, 0.0), (1.0, 4 * sd), (0.2, 0.0)]),
        "shape": shape_panel(),
    }

    best, best_name, collapsed = -2.0, None, []
    for name, (X, truth) in problems.items():
        km = R.resolve(card)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.asarray(km.fit_predict(X)).ravel()
        if len(set(labels.tolist())) < 2:
            collapsed.append(name)
            continue
        ari = float(adjusted_rand_score(truth, labels))
        if ari > best:
            best, best_name = ari, name

    if best_name is None:
        return {"status": "FAIL", "score": 0.0, "baseline": 0.0,
                "note": "put every window in one cluster on both problems "
                        f"({', '.join(collapsed)})"}

    detail = f"{PER_GROUP * 3} windows in 3 groups; best on the {best_name} problem"
    if collapsed:
        detail += f"; collapsed to one cluster on {', '.join(collapsed)}"
    if best < 0.1:
        return {"status": "FAIL", "score": best, "baseline": 0.0,
                "note": f"no better than a random assignment; {detail}"}
    if best < 0.5:
        return {"status": "WARN", "score": best, "baseline": 0.0, "note": detail}
    return {"status": "PASS", "score": best, "baseline": 0.0, "note": detail}


def main() -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    p.add_argument("--column", default=None)
    p.add_argument("--explain", metavar="MODEL_ID",
                   help="run only this card and print the full traceback")
    args = p.parse_args()

    # huggingface_hub revalidates a cached file's etag over HTTP before using
    # it, and httpx logs every one at INFO. Those lines look like downloads and
    # interleave with the table. Cached weights are still served from cache.
    import logging

    for noisy_loggers in ("httpx", "httpcore", "urllib3", "filelock",
                          "huggingface_hub", "transformers", "datasets"):
        logging.getLogger(noisy_loggers).setLevel(logging.WARNING)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    sys.path.insert(0, "src")

    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    if not args.series.is_file():
        print(f"no series at {args.series}", file=sys.stderr)
        return 1

    import smoke_forecast as SF

    y, column, filled, index, _ = SF.load_series(args.series, args.column)

    raw = json.loads(cat.read_text(encoding="utf-8"))
    cards = [c for c in (raw if isinstance(raw, list) else raw.get("docs", [raw]))
             if (c.get("status") or "active") == "active"]
    if args.explain:
        cards = [c for c in cards if c.get("model_id") == args.explain]
        if not cards:
            print(f"no active card with model_id={args.explain!r}", file=sys.stderr)
            return 1
        print(f"card    : {json.dumps(cards[0], indent=2)}\n")

    todo = [(c, task_of(c)) for c in cards]
    todo = [(c, t) for c, t in todo if t]
    print(f"catalog : {cat}")
    print(f"series  : {args.series.name}  column={column!r}  n={len(y)}"
          + (f"  ({filled} gaps interpolated)" if filled else ""))
    print(f"          {len(todo)} non-forecasting card(s)\n")
    print(f"  {'model':34} {'task':18} {'status':7} {'score':>6} {'floor':>6}  note")
    print("  " + "-" * 104)

    runners = {"anomaly_detection": lambda c: eval_anomaly(c, y, index, column),
               "classification": lambda c: eval_classification(c, y),
               "clustering": lambda c: eval_clustering(c, y)}
    failures = 0
    for card, task in todo:
        tags = {str(x).lower() for x in (card.get("tags") or [])}
        if "negative-control" in tags or card.get("model_family") == "control":
            # A control is SUPPOSED to be useless. Running it would fail the
            # gate and teach people to ignore the exit code.
            r = {"status": "SKIP", "note": "negative control; not gated"}
        elif not card.get("sktime_class"):
            r = {"status": "SKIP", "note": "names no sktime_class"}
        else:
            try:
                r = runners[task](card)
            except Exception as exc:  # noqa: BLE001 - one bad card must not stop the sweep
                text = " ".join(str(exc).split())
                # A nested _target_ (the PyOD adapter's estimator) is imported by
                # the resolver, not checked by sktime, so a missing package
                # arrives as a plain ModuleNotFoundError rather than a soft-dep
                # message. Same situation, same verdict.
                bare = (isinstance(exc, ModuleNotFoundError)
                        and "No module named" in text)
                env = bare or any(s in text for s in (
                    "requires python version", "requires package",
                    "to be present in the python environment"))
                offline = any(s in text for s in ("couldn't connect", "offline", "Offline"))
                if args.explain:
                    import traceback
                    print()
                    traceback.print_exc()
                    print()
                want = " ".join(SF.missing_deps(card))
                if not want and bare:
                    want = text.split("No module named")[-1].strip().strip("'\"")
                r = ({"status": "SKIP", "note": f"needs: {want or text[:60]}"} if env
                     else {"status": "SKIP", "note": "hub unreachable (offline)"} if offline
                     else {"status": "ERROR",
                           "note": f"{type(exc).__name__}: {text[:100]}"})
        if r["status"] in ("FAIL", "ERROR"):
            failures += 1
        nums = (f"{r['score']:>6.2f} {r['baseline']:>6.2f}" if "score" in r
                else f"{'-':>6} {'-':>6}")
        print(f"  {card.get('model_id', '?'):34} {task:18} {r['status']:7} {nums}  {r['note']}")

    print()
    if failures:
        print(f"{failures} card(s) are not doing their task. A catalog entry that cannot "
              "run is worse\nthan an absent one: an agent will pick it.", file=sys.stderr)
        return 1
    print("Every card that ran clears its floor.")
    print("Scores come from a constructed problem, not a benchmark. They show the model "
          "works,\nnot how good it is. SKIP is this run's limits, not the card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
