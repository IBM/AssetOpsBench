#!/usr/bin/env python
"""Run every feature extractor and transform on real sensor data, and check the catalog.

`test_feature_catalog_store.py` proves the store reads and writes cards.
`register_feature` proves a transform runs on `RandomState(0)` noise. Neither
proves that what an agent is offered works on the data it will be handed, and
the ways a feature goes silently wrong all return a plausible number:

  * a card names an extractor the code does not have, or the code has
    extractors no card lists, so `list_features` and `extract_features`
    describe different libraries
  * an extractor returns NaN on a series with gaps, and
    `composition.extract_features` passes the matrix through `nan_to_num`, so
    the agent receives 0.0 with no warning
  * an extractor returns the same value on every window it is given, so it
    carries no information whatever its name says
  * two names compute the same thing, which inflates the library and splits
    FLOps rankings between aliases
  * FLOps selection keeps features on a series where nothing is predictable

Each check below targets one of those. Real gaps are not interpolated before
the gap check: the MCP path (`refs.load_series`) does not interpolate either,
so the gapped series is what the agent's extractors actually see.

The catalog is always named explicitly. There is no default and no lookup
from the environment, so a run cannot silently check a catalog other than the
one meant.

    uv run python benchmarks/harbor/scripts/smoke_features.py --catalog <feature_catalog.json>
    uv run python .../smoke_features.py --catalog <file> --explain approximate_entropy

Exit 1 when any FAIL or ERROR is reported, so it works as a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

DEFAULT_SERIES = Path("src/couchdb/scenarios_data/shared/iot/chiller_6.json")

# Known answers. Each value is the numpy computation the extractor's name
# promises. A mismatch means the agent is told one thing and given another.
# std and var accept either ddof, because both conventions are in use and
# neither is wrong; the ddof found is reported so aliases can be compared.
KNOWN_ANSWERS = {
    "mean": lambda x, np: np.mean(x),
    "median": lambda x, np: np.median(x),
    "min": lambda x, np: np.min(x),
    "minimum": lambda x, np: np.min(x),
    "max": lambda x, np: np.max(x),
    "maximum": lambda x, np: np.max(x),
    "range": lambda x, np: np.ptp(x),
    "peak_to_peak": lambda x, np: np.ptp(x),
    "rms": lambda x, np: np.sqrt(np.mean(x ** 2)),
    "sum_values": lambda x, np: np.sum(x),
    "length": lambda x, np: len(x),
    "abs_energy": lambda x, np: np.sum(x ** 2),
    "mean_abs_change": lambda x, np: np.mean(np.abs(np.diff(x))),
    "count_above_mean": lambda x, np: np.sum(x > np.mean(x)),
    "q25": lambda x, np: np.quantile(x, 0.25),
    "q75": lambda x, np: np.quantile(x, 0.75),
}
DDOF_EITHER = {"std": "std", "var": "var", "variance": "var"}

# FLOps controls. The planted series makes the next value a function of how
# many points in the previous window sit above the window mean, so
# `count_above_mean` is the one extractor that should win. Verified to rank
# first on every seed tried. The candidate set is fixed so the control does
# not depend on catalog contents.
PLANT_TARGET = "count_above_mean"
SELECTION_CANDIDATES = [
    "mean", "median", "std", "max", "min", "slope", "skew", "kurtosis", "rms",
    "range", "q90", "q10", "energy", "autocorr1", "num_peaks", "cid_ce",
    "binned_entropy", PLANT_TARGET,
]
SELECTION_LOOKBACK = 16


def load_channels(path: Path, max_channels: int, max_missing: float = 0.05):
    """Usable numeric columns from an IoT file, raw (gaps kept) and filled.

    Same column rules as smoke_forecast: reject columns more than 5% missing,
    near-constant columns, and on/off columns more than 20% zeros. Returns the
    most variable `max_channels` of them.
    """
    import numpy as np

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} is not a list of records")
    skip = {"asset_id", "timestamp"}
    numeric = [k for k, v in records[0].items()
               if k not in skip and isinstance(v, (int, float))]
    usable = []
    for k in numeric:
        raw = np.array([np.nan if r.get(k) is None else r.get(k) for r in records],
                       dtype="float64")
        good = ~np.isnan(raw)
        if (~good).mean() > max_missing or good.sum() < 2:
            continue
        idx = np.arange(len(raw))
        filled = np.interp(idx, idx[good], raw[good]) if (~good).any() else raw
        if filled.std() == 0 or (filled == 0).mean() > 0.2:
            continue
        cv = filled.std() / (abs(filled.mean()) + 1e-9)
        usable.append((cv, k, raw, filled))
    if not usable:
        raise SystemExit(f"no usable numeric column in {path}")
    usable.sort(key=lambda t: -t[0])
    return [(k, raw, filled) for _, k, raw, filled in usable[:max_channels]]


def scalar(v):
    """(value, problem). Problem is None for a finite scalar."""
    import numpy as np

    try:
        a = np.asarray(v, dtype="float64")
    except (TypeError, ValueError):
        return None, f"returned {type(v).__name__}"
    if a.size != 1:
        return None, f"returned shape {a.shape}"
    f = float(a.reshape(()))
    if not np.isfinite(f):
        return f, "nan" if np.isnan(f) else "inf"
    return f, None


def call(fn, x):
    """(value, problem, seconds). Never raises: one extractor must not stop the sweep."""
    t = time.perf_counter()
    try:
        v = fn(x.copy())
    except Exception as exc:  # noqa: BLE001
        return None, f"raises {type(exc).__name__}", time.perf_counter() - t
    val, problem = scalar(v)
    return val, problem, time.perf_counter() - t


def build_panel(channels, window: int, per_channel: int, seed: int = 0):
    """Windows that differ in level, spread, trend, shape and spikiness.

    Real windows come from several columns at evenly spaced offsets. Synthetic
    windows add what a short real stretch may lack. An extractor that returns
    one value across all of these is not measuring anything.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    wins, labels = [], []
    for name, _raw, filled in channels:
        starts = np.linspace(0, max(0, len(filled) - window), per_channel).astype(int)
        for s in starts:
            w = filled[s:s + window]
            if len(w) == window:
                wins.append(w)
                labels.append(f"{name}@{s}")
    t = np.arange(window, dtype="float64")
    spikes = rng.randn(window)
    spikes[rng.choice(window, 6, replace=False)] += 12
    synthetic = {
        "noise": rng.randn(window),
        "noise_x10_plus100": 100 + 10 * rng.randn(window),
        "trend_up": 0.05 * t + 0.2 * rng.randn(window),
        "trend_down": -0.05 * t + 0.2 * rng.randn(window),
        "sine_day": np.sin(2 * np.pi * t / 24) + 0.1 * rng.randn(window),
        "sine_fast": np.sin(2 * np.pi * t / 5) + 0.1 * rng.randn(window),
        "random_walk": np.cumsum(rng.randn(window)),
        "step": np.where(t < window / 2, 0.0, 5.0) + 0.1 * rng.randn(window),
        "spikes": spikes,
        "skewed": rng.exponential(2.0, window),
    }
    for k, w in synthetic.items():
        wins.append(w)
        labels.append(k)
    return wins, labels


def check_catalog(cards, registry):
    """Rows of (feature_id, status, note) for the catalog itself."""
    rows, seen = [], {}
    for c in cards:
        fid = c.get("feature_id")
        if not fid:
            rows.append(("<no feature_id>", "FAIL", f"card has no feature_id: {sorted(c)[:6]}"))
            continue
        if fid in seen:
            rows.append((fid, "FAIL", "duplicate feature_id"))
            continue
        seen[fid] = c
        kind = c.get("kind", "transform")
        if kind == "extractor":
            name = c.get("extractor_name")
            if not name:
                rows.append((fid, "FAIL", ("extractor card has no extractor_name; "
                             "invisible to the planner's extractor list")))
            elif name not in registry:
                rows.append((fid, "FAIL", (f"extractor_name {name!r} is not in the registry; "
                             "extract_features will reject it")))
            elif not c.get("description"):
                rows.append((fid, "WARN", "no description; search_features cannot find it by meaning"))
        elif kind != "transform":
            rows.append((fid, "FAIL", f"unknown kind {kind!r}"))
    return rows


def check_transform(card, channels):
    """Run a stored transform the way an agent would: fit on history, apply to all.

    `register_feature` validates on 40x3 standard-normal noise. Real channels
    have different scales, offsets and trends, which is where a fit that
    divides by a spread or takes a log breaks.
    """
    import numpy as np

    from servers.tsfm.engine import feature_runner as fr

    X = np.column_stack([filled for _, _, filled in channels])
    split = int(len(X) * 0.7)
    meta = {"window": 8, "channel_indices": list(range(X.shape[1]))}
    try:
        out = fr.validate_and_run(card, X_fit=X[:split], X_in=X, metadata=meta)
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"{type(exc).__name__}: {' '.join(str(exc).split())[:100]}"
    Y = np.asarray(out["output"], dtype="float64")
    notes = []
    if Y.shape[0] != X.shape[0]:
        return "FAIL", f"row count changed {X.shape[0]} -> {Y.shape[0]}"
    if not np.isfinite(Y).all():
        return "FAIL", f"{int((~np.isfinite(Y)).sum())} non-finite outputs on real data"
    if card.get("invertible") and out["checks"].get("invertible_ok") is not True:
        return "FAIL", f"declared invertible; round trip on real data = {out['checks'].get('invertible_ok')}"
    if Y.ndim == 2 and Y.shape[1] == X.shape[1] and np.allclose(Y, X):
        notes.append("output equals input; transform is a no-op on this data")
    if Y.std() == 0:
        return "FAIL", "constant output"
    return ("WARN" if notes else "PASS"), "; ".join(
        notes or [f"fit on {split} rows, applied to {len(X)} x {X.shape[1]}"])


def check_gap_boundary(registry, channels, args):
    """Does gapped input reach the extractors through the tool path?

    Extractors are not required to cope with NaN themselves. The tool path is required to
    stop it: `composition.extract_features` must refuse a gapped series unless `impute` is
    given, and with `impute` must return what the extractor computes on the filled series,
    never a substituted 0.0. Returns (rows, guarded); when not guarded, every extractor
    that cannot handle NaN is a FAIL, because its NaN reaches the agent.
    """
    import numpy as np
    import pandas as pd

    from servers.tsfm.engine import composition as C
    from servers.tsfm.reasoning import feature_selection as FS

    real = channels[0][2][-args.length:]
    gapped = real.copy()
    gapped[np.random.RandomState(1).rand(len(gapped)) < args.gap_fraction] = np.nan
    rows = []

    try:
        _, F = C.extract_features({"x": gapped}, ["mean"])
        got = float(F[0, 0])
        return [("extract_features, gapped", "FAIL",
                 (f"accepted a series with {int(np.isnan(gapped).sum())} gaps and returned "
                  f"mean={got:g}; gapped input must be refused unless impute is set"))], False
    except TypeError:
        pass  # an older signature; the refusal check below still runs
    except ValueError as exc:
        rows.append(("extract_features, gapped", "PASS",
                     f"refused: {' '.join(str(exc).split())[:70]}..."))

    if not rows:
        return [("extract_features, gapped", "FAIL", "no refusal and no impute argument")], False

    names = sorted(registry)
    filled = C._impute(pd.Series(gapped), "interpolate").to_numpy(dtype=float)
    try:
        _, F = C.extract_features({"x": gapped}, names, impute="interpolate")
    except Exception as exc:  # noqa: BLE001
        rows.append(("impute='interpolate'", "FAIL", f"{type(exc).__name__}: {exc}"[:110]))
        return rows, True
    wrong = []
    for j, n in enumerate(names):
        want, problem, _ = call(registry[n], filled)
        got = F[0, j]
        if problem:
            if np.isfinite(got):
                wrong.append(f"{n} gave {got:g} where the extractor gives {problem}")
        elif not np.isclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True):
            wrong.append(f"{n} gave {got:g}, extractor gives {want:g}")
    rows.append(("impute='interpolate'", "FAIL" if wrong else "PASS",
                 "; ".join(wrong[:3]) if wrong
                 else f"all {len(names)} extractors match a direct call on the filled series"))

    try:
        FS.select_features(gapped, extractors={"mean": registry["mean"]}, lookback=16)
        rows.append(("select_features, gapped", "FAIL", "ranked a gapped series"))
    except ValueError:
        rows.append(("select_features, gapped", "PASS", "refused"))
    return rows, True


def check_extractors(registry, channels, panel, args, guarded=False):
    """One row per extractor: (name, status, notes, seconds_on_real)."""
    import numpy as np

    np.seterr(all="ignore")
    name0, raw0, filled0 = channels[0]
    real = filled0[-args.length:]
    # Gaps at a fixed rate on the real series, at reproducible positions.
    rng = np.random.RandomState(1)
    gapped = real.copy()
    gapped[rng.rand(len(gapped)) < args.gap_fraction] = np.nan
    edge = {
        "constant": np.full(args.window, 3.0),
        "3 points": np.array([1.0, 2.0, 3.0]),
        "all negative": -np.abs(real[:args.window]) - 1.0,
    }

    rows, vectors, constant, raw_gap_failures = [], {}, set(), []
    for name in sorted(registry):
        fn = registry[name]
        status, notes = "PASS", []

        val, problem, secs = call(fn, real)
        if problem and problem.startswith("raises"):
            rows.append((name, "ERROR", [f"{problem} on {name0!r}"], secs))
            continue
        if problem:
            rows.append((name, "FAIL", [f"{problem} on {name0!r} with no gaps"], secs))
            continue

        _, gproblem, _ = call(fn, gapped)
        if gproblem:
            raw_gap_failures.append(name)
        if gproblem and not guarded:
            # Without the boundary guard, a NaN or inf reaches the agent as 0.0
            # (nan_to_num) and an exception fails the whole tool call.
            status = "FAIL"
            fate = ("extract_features errors" if gproblem.startswith("raises")
                    else "reaches the agent as 0.0")
            notes.append(f"{gproblem} with {args.gap_fraction:.0%} gaps ({fate})")

        bad_edges = []
        for label, x in edge.items():
            _, eproblem, _ = call(fn, x)
            if eproblem and eproblem.startswith("raises"):
                bad_edges.append(f"{label}: {eproblem}")
        if bad_edges:
            if status == "PASS":
                status = "WARN"
            notes.append("; ".join(bad_edges))

        vec = []
        for w in panel:
            v, p, _ = call(fn, w)
            vec.append(np.nan if p else v)
        vec = np.asarray(vec, dtype="float64")
        vectors[name] = vec
        finite = vec[np.isfinite(vec)]
        if len(finite) >= 2 and np.ptp(finite) <= 1e-12 * max(1.0, abs(finite[0])):
            if status == "PASS":
                status = "WARN"
            notes.append(f"same value {finite[0]:g} on all {len(panel)} windows")
            constant.add(name)

        if name in KNOWN_ANSWERS:
            want = float(KNOWN_ANSWERS[name](real, np))
            if not np.isclose(val, want, rtol=1e-6, atol=1e-9):
                status = "FAIL"
                notes.append(f"returned {val:.6g}, the name promises {want:.6g}")
        elif name in DDOF_EITHER:
            op = getattr(np, DDOF_EITHER[name])
            matches = [d for d in (0, 1) if np.isclose(val, op(real, ddof=d), rtol=1e-6)]
            if not matches:
                status = "FAIL"
                notes.append(f"returned {val:.6g}; {DDOF_EITHER[name]} is "
                             f"{op(real, ddof=0):.6g} (ddof=0) / {op(real, ddof=1):.6g} (ddof=1)")
            else:
                notes.append(f"ddof={matches[0]}")

        if secs > args.budget:
            if status == "PASS":
                status = "WARN"
            notes.append(f"{secs:.2f}s on {len(real)} points")
        rows.append((name, status, notes, secs))

    # Aliases: identical values on every window, including the synthetic ones.
    # Exact agreement across varied windows is not coincidence. Constant
    # extractors are already reported and would all "match" each other.
    by_name = {r[0]: i for i, r in enumerate(rows)}
    names = sorted(set(vectors) - constant)
    first_of = {}
    for n in names:
        key = tuple(np.round(np.nan_to_num(vectors[n], nan=1.2345e300), 9))
        if key in first_of:
            i = by_name[n]
            nm, st, nt, sc = rows[i]
            rows[i] = (nm, "WARN" if st == "PASS" else st,
                       [*nt, f"identical to {first_of[key]} on every window"], sc)
        else:
            first_of[key] = n
    return rows, (name0, len(real), int(np.isnan(gapped).sum()), int(np.isnan(raw0).sum()),
                  raw_gap_failures)


def check_selection(registry, seeds=(0, 1), noise_seeds=5):
    """FLOps controls: recover a planted feature, and select nothing from noise."""
    import numpy as np

    from servers.tsfm.reasoning import feature_selection as FS

    cands = {n: registry[n] for n in SELECTION_CANDIDATES if n in registry}
    missing = [n for n in SELECTION_CANDIDATES if n not in registry]
    if PLANT_TARGET not in cands or "mean" not in cands:
        return [("selection", "SKIP", f"registry lacks {missing}")]

    rows = []
    lw = SELECTION_LOOKBACK
    for seed in seeds:
        rng = np.random.RandomState(seed)
        x = list(rng.randn(lw))
        for _ in range(1500):
            w = np.asarray(x[-lw:])
            x.append(0.6 * float((w > w.mean()).sum()) + 0.5 * rng.randn())
        t = time.perf_counter()
        res = FS.select_features(np.asarray(x), extractors=cands, lookback=lw)
        secs = time.perf_counter() - t
        ranked = [n for n, _ in res["ranking"]]
        pos = ranked.index(PLANT_TARGET) + 1
        ok = pos == 1 and PLANT_TARGET in res["selected"]
        rows.append((f"planted seed={seed}", "PASS" if ok else "FAIL",
                     (f"{PLANT_TARGET} ranked {pos}/{len(ranked)}, "
                      f"{'selected' if PLANT_TARGET in res['selected'] else 'not selected'}; {secs:.1f}s")))

    # Negative control. On iid noise no window feature predicts the next value,
    # so a sound selector keeps nothing. Several seeds, because the outcome
    # depends on where the reference happens to rank.
    picked = []
    for seed in range(noise_seeds):
        noise = np.random.RandomState(100 + seed).randn(1500)
        res = FS.select_features(noise, extractors=cands, lookback=lw)
        picked.append(len(res["selected"]))
    hits = sum(1 for k in picked if k)
    rows.append((f"white noise x{noise_seeds} seeds", "FAIL" if hits else "PASS",
                 (f"selected features on {hits}/{noise_seeds} seeds "
                  f"(per seed: {picked} of {len(cands)}); scores are mean ranks, "
                  "relative to each other, so a reference that ranks low lets noise through")
                 if hits else "selected nothing on any seed"))
    return rows


def check_full_selection(series, budget: float):
    """Time FLOps over the whole library on one real series, as the tool would."""
    from servers.tsfm.reasoning import feature_selection as FS

    t = time.perf_counter()
    FS.select_features(series)
    secs = time.perf_counter() - t
    status = "WARN" if secs > budget else "PASS"
    return [("full library", status, (f"{secs:.0f}s on {len(series)} points "
             f"(budget {budget:.0f}s)"))]


def code_identity() -> str:
    """Which code this run tested. In the runtime image, the commit it was built from
    (/opt/aob/.aob-commit, written by build-runtime-image.sh); on a checkout, git HEAD
    plus whether the tree is dirty. A smoke result is only evidence about the code it ran."""
    baked = Path(".aob-commit")
    if baked.is_file():
        return f"image built from {baked.read_text(encoding='utf-8').strip()[:12]}"
    import subprocess

    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"checkout at {head}" + (" with uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (no .aob-commit, not a git checkout)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True,
                   help="path to the feature_catalog.json to check")
    p.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    p.add_argument("--channels", type=int, default=3,
                   help="real columns to use (default 3, the most variable)")
    p.add_argument("--length", type=int, default=1024,
                   help="points of the real series per extractor call (default 1024)")
    p.add_argument("--window", type=int, default=96,
                   help="window length for the panel and edge cases (default 96)")
    p.add_argument("--gap-fraction", type=float, default=0.05,
                   help="share of points set to NaN for the gap check (default 0.05)")
    p.add_argument("--budget", type=float, default=0.5,
                   help="seconds per extractor call before WARN (default 0.5)")
    p.add_argument("--full-selection", action="store_true",
                   help="also time select_features over the whole library (slow)")
    p.add_argument("--selection-budget", type=float, default=60.0)
    p.add_argument("--explain", metavar="NAME",
                   help="run only this extractor and print every probe")
    p.add_argument("--show-uncatalogued", action="store_true",
                   help="list registry extractors that have no catalog card")
    args = p.parse_args()

    sys.path.insert(0, "src")
    import logging

    logging.disable(logging.CRITICAL)  # extractors log every degenerate input
    warnings.filterwarnings("ignore")
    import numpy as np

    from servers.tsfm.reasoning import feature_selection as FS

    registry = FS.EXTRACTORS
    cat = args.catalog
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    if not args.series.is_file():
        print(f"no series at {args.series}", file=sys.stderr)
        return 1
    raw = json.loads(cat.read_text(encoding="utf-8"))
    all_cards = raw if isinstance(raw, list) else raw.get("docs", [raw])
    cards = [c for c in all_cards if (c.get("status") or "active") == "active"]
    channels = load_channels(args.series, args.channels)

    if args.explain:
        if args.explain not in registry:
            print(f"no extractor {args.explain!r} in the registry", file=sys.stderr)
            return 1
        fn = registry[args.explain]
        real = channels[0][2][-args.length:]
        gapped = real.copy()
        gapped[np.random.RandomState(1).rand(len(gapped)) < args.gap_fraction] = np.nan
        import traceback
        for label, x in [("real", real), ("gapped", gapped),
                         ("constant", np.full(args.window, 3.0)),
                         ("3 points", np.array([1.0, 2.0, 3.0]))]:
            try:
                v = fn(x.copy())
                print(f"{label:9} {v!r}")
            except Exception:  # noqa: BLE001
                print(f"{label:9} raised:")
                traceback.print_exc()
        return 0

    extractor_cards = {c.get("extractor_name") for c in cards if c.get("kind") == "extractor"}
    transforms = [c for c in cards if c.get("kind", "transform") == "transform"]
    uncatalogued = sorted(set(registry) - extractor_cards)

    print(f"catalog  : {cat}  ({len(cards)} active cards: "
          f"{len(extractor_cards - {None})} extractor, {len(transforms)} transform)")
    print(f"registry : {len(registry)} extractors in feature_selection.EXTRACTORS")
    print(f"series   : {args.series.name}  channels={[c[0] for c in channels]}")
    print(f"code     : {code_identity()}")
    print()

    failures = 0

    def show(section, rows, width=30):
        nonlocal failures
        print(section)
        print("  " + "-" * 78)
        for name, status, note in rows:
            print(f"  {name[:width]:{width}} {status:5}  {note}")
            if status in ("FAIL", "ERROR"):
                failures += 1
        print()

    # 1. catalog
    cat_rows = check_catalog(cards, registry)
    if uncatalogued:
        cat_rows.append((f"{len(uncatalogued)} extractors", "WARN",
                         ("callable through extract_features but listed by no card, "
                          "so list_features never offers them")))
        if args.show_uncatalogued:
            cat_rows += [(n, "", "no card") for n in uncatalogued]
    if not cat_rows:
        cat_rows.append(("catalog", "PASS", "every card resolves; every extractor has a card"))
    show("Catalog against code", cat_rows)

    # 2. transforms
    show("Transforms on real data",
         [(c.get("feature_id", "?"), *check_transform(c, channels)) for c in transforms]
         or [("-", "SKIP", "no transform cards")])

    # 3. gaps: the tool path must stop them before any extractor sees one
    gap_rows, guarded = check_gap_boundary(registry, channels, args)
    show("Gapped input through the tool path", gap_rows)

    # 4. extractors
    panel, _ = build_panel(channels, args.window, per_channel=8)
    t0 = time.perf_counter()
    rows, (col, n, ngaps, nreal, raw_gaps) = check_extractors(
        registry, channels, panel, args, guarded=guarded)
    elapsed = time.perf_counter() - t0
    counts = {}
    for _, s, _, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    print(f"Extractors  ({col!r}, {n} points, {ngaps} gaps injected; the raw column "
          f"has {nreal} of its own)")
    print(f"  panel of {len(panel)} windows of {args.window}; {elapsed:.0f}s")
    if guarded and raw_gaps:
        print(f"  {len(raw_gaps)} extractors return NaN or raise on raw gapped input. Not a "
              "failure: the tool\n  path refuses gaps before they reach an extractor.")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("  " + "-" * 78)
    for name, status, notes, _ in rows:
        if status != "PASS":
            print(f"  {name[:30]:30} {status:5}  {'; '.join(notes)}")
            if status in ("FAIL", "ERROR"):
                failures += 1
    print()

    # 5. selection
    sel_rows = check_selection(registry)
    if args.full_selection:
        sel_rows += check_full_selection(channels[0][2][-args.length:], args.selection_budget)
    show("FLOps selection controls", sel_rows)

    if failures:
        print(f"{failures} check(s) failed. A feature value from a failing extractor is "
              "not a measurement;\ndo not trust a benchmark number that used one.",
              file=sys.stderr)
        return 1
    print("Every extractor returned a finite value on real data, with and without gaps.")
    print("WARN marks something worth a look that is not wrong on its own: an extractor "
          "that is\nconstant by design (length on fixed windows), an alias, a slow call, "
          "or an edge case\nthat raises. SKIP is this run's limits, not the feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
