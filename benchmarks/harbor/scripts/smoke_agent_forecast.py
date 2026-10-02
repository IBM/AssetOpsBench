#!/usr/bin/env python
"""Forecast with every catalog model the way an agent does: through the TSFM MCP tools.

`smoke_forecast.py` builds each estimator directly and scores it against naive.
An agent never does that. It calls `list_models` / `find_models` /
`describe_models` / `resolve_model`, then `run_recipe` with
`{"estimator": {"model_id": ...}, "fh": [...]}`, and reads `backtest_score`,
`training_regime` and the message. This script makes those calls and holds each
answer to the same contract as `smoke_agent_features.py`:

    the tool either does what was asked, or returns an error that names the cause.

On top of that, for every run that succeeds:

  * the score must be finite, and the forecast must beat or match naive on the
    same holdout (worse than 2x naive is a FAIL, worse than 1x a WARN);
  * a card the catalog says is zero-shot must run zero-shot (a run that trained
    on the series is reported as that, not as a zero-shot score);
  * `resolve_model` saying "resolvable" must mean `run_recipe` runs;
  * a series with gaps must either forecast to a finite score or be refused
    with a message that names the gaps.

Naive is computed here on exactly the holdout `run_recipe` scores. That matters:
a zero-shot recipe is scored on one final holdout of `len(fh)` points, a
recipe that fits is scored over many expanding folds, so their `backtest_score`
values are not comparable with each other. The header prints both protocols.

Calls go through FastMCP's `call_tool`, against an in-memory store seeded from
the catalog you pass. Run it in the runtime image so the preloaded weights are used:

    uv run python benchmarks/harbor/scripts/smoke_agent_forecast.py --model-catalog <model_catalog.json>

Exit 1 when any FAIL is reported, so it works as a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

DEFAULT_SERIES = Path("src/couchdb/scenarios_data/shared/iot/chiller_6.json")
FORECAST_TASK = "tsfm_forecasting"


def load_series(path: Path, column: str | None):
    """The most variable non-intermittent column (or the one named), raw and filled,
    plus one day in steps for the seasonal baseline."""
    import numpy as np
    import pandas as pd

    records = json.loads(path.read_text(encoding="utf-8"))
    stamps = pd.to_datetime([r.get("timestamp") for r in records], errors="coerce")
    step = pd.Series(stamps).diff().median() if not stamps.isna().any() else None
    season = max(2, round(86400 / step.total_seconds())) if step is not None else 24
    skip = {"asset_id", "timestamp"}
    best = None
    for k, v in records[0].items():
        if k in skip or not isinstance(v, (int, float)) or (column and k != column):
            continue
        raw = np.array([np.nan if r.get(k) is None else r.get(k) for r in records], float)
        gaps = int(np.isnan(raw).sum())
        if gaps > 0.05 * len(raw) or np.nanstd(raw) == 0 or (raw == 0).mean() > 0.2:
            continue
        cv = np.nanstd(raw) / (abs(np.nanmean(raw)) + 1e-9)
        if best is None or cv > best[0]:
            best = (cv, k, raw)
    if best is None:
        raise SystemExit(f"no usable column in {path}")
    _, name, raw = best
    idx = np.arange(len(raw))
    good = ~np.isnan(raw)
    filled = np.interp(idx, idx[good], raw[good])
    freq = f"{int(step.total_seconds())}s" if step is not None else "h"
    return name, raw, filled, season, freq


def naive_holdout_mae(y, h: int, season: int) -> float:
    """MAE of the better of last-value and seasonal naive on the final h points, the
    holdout `_backtest_zero_shot` scores a zero-shot recipe on."""
    import numpy as np

    train, test = y[:-h], y[-h:]
    last = np.full(h, train[-1])
    seas = np.resize(train[-season:][:h], h) if len(train) >= season else last
    return float(min(np.mean(np.abs(test - last)), np.mean(np.abs(test - seas))))


def card_regime(card: dict) -> str | None:
    return card.get("training_regime") or (card.get("params") or {}).get("fit_strategy")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-catalog", type=Path, required=True,
                   help="path to the model_catalog.json to serve to the agent")
    p.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    p.add_argument("--column", default=None, help="sensor column (default: the most variable)")
    p.add_argument("--fh", type=int, default=24, help="forecast horizon in steps (default 24)")
    p.add_argument("--models", nargs="*", default=None, help="only these model ids")
    p.add_argument("--show-errors", action="store_true",
                   help="print the full error text the agent receives")
    args = p.parse_args()

    if not args.model_catalog.is_file():
        print(f"no model catalog at {args.model_catalog}", file=sys.stderr)
        return 1
    if not args.series.is_file():
        print(f"no series at {args.series}", file=sys.stderr)
        return 1

    os.environ["TSFM_STORE"] = "memory"   # before the server module builds its stores
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    sys.path.insert(0, "src")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import logging

    logging.disable(logging.CRITICAL)
    warnings.showwarning = lambda *a, **k: None
    from smoke_agent_features import Agent, error_of, names_cause
    from smoke_features import code_identity

    from servers.tsfm import main as server
    from servers.tsfm.io import refs
    from servers.tsfm.stores import model_store

    raw = json.loads(args.model_catalog.read_text(encoding="utf-8"))
    cards = [c for c in (raw if isinstance(raw, list) else raw.get("docs", [raw]))
             if (c.get("status") or "active") == "active"]
    for c in cards:
        doc = dict(c)
        doc.setdefault("_id", f"model:{doc['model_id']}")
        server._STORE.put(model_store.collection_name(), doc)
    by_id = {c["model_id"]: c for c in cards}
    agent = Agent(server)

    column, raw_y, y, season, freq = load_series(args.series, args.column)
    clean = refs.materialize_iot(y, asset_id="agent_fc_clean", freq=freq)
    gapped = refs.materialize_iot(raw_y, asset_id="agent_fc_gapped", freq=freq)
    h = args.fh
    fh = list(range(1, h + 1))
    naive_h = naive_holdout_mae(y, h, season)

    print(f"catalog  : {args.model_catalog}  ({len(cards)} active cards)")
    print(f"series   : {args.series.name}  column={column!r}  n={len(y)}  "
          f"({int(sum(map(math.isnan, raw_y)))} real gaps)  seasonal lag {season}")
    print(f"code     : {code_identity()}")
    print(f"horizon  : fh=1..{h}; naive MAE on the final {h}-point holdout = {naive_h:.4g}")
    print()

    failures = 0

    def show(section, rows):
        nonlocal failures
        print(section)
        print("  " + "-" * 78)
        for name, status, note, err in rows:
            print(f"  {name[:34]:34} {status:5}  {note}")
            if args.show_errors and err:
                print(f"  {'':34}        agent sees: {' '.join(str(err).split())[:220]}")
            if status == "FAIL":
                failures += 1
        print()

    # ------------------------------------------------------------------ discovery
    rows = []
    want = sorted(c["model_id"] for c in cards if FORECAST_TASK in (c.get("task_ids") or []))
    r = agent.call("list_models", {"task_id": FORECAST_TASK})
    listed = sorted(m.get("model_id") for m in (r.get("models") or []))
    size = len(json.dumps(r))
    rows.append(("list_models(tsfm_forecasting)",
                 "PASS" if listed == want else "FAIL",
                 f"{len(listed)} cards, response {size:,} chars (~{size // 4:,} tokens)"
                 + ("" if listed == want else f"; catalog has {len(want)}"), error_of(r)))
    if size > 40_000:
        rows.append(("list_models response size", "WARN",
                     "large enough to crowd the agent's context; find_models is the shortlist",
                     None))

    r = agent.call("find_models", {"task_id": FORECAST_TASK, "top_k": 5})
    found = [m.get("model_id") for m in (r.get("models") or [])]
    rows.append(("find_models(top_k=5)",
                 "PASS" if found and set(found) <= set(listed) else "FAIL",
                 f"{found}", error_of(r)))

    r = agent.call("describe_models", {"model_ids": listed[:5]})
    got = [m.get("model_id") for m in (r.get("models") or [])]
    rows.append(("describe_models(5 listed ids)",
                 "PASS" if len(got) == len(listed[:5]) and not r.get("unknown") else "FAIL",
                 f"found {len(got)}/{len(listed[:5])}, unknown={r.get('unknown')}",
                 error_of(r)))

    r = agent.call("run_recipe", {"dataset_path": clean, "timestamp_column": "timestamp",
                                  "target_columns": ["value"],
                                  "recipe": {"estimator": {"model_id": "not_a_model"},
                                             "fh": fh}})
    err = error_of(r)
    rows.append(("run_recipe, unknown model_id",
                 "PASS" if names_cause(err, "not_a_model") else "FAIL",
                 "refused, names it" if names_cause(err, "not_a_model")
                 else ("ran" if not err else "error does not name the model"), err))
    show("Discovery (what the agent reads before choosing)", rows)

    # ------------------------------------------------------------- every model
    ids = [m for m in listed if not args.models or m in args.models]
    rows = []
    ran = []
    for mid in ids:
        card = by_id.get(mid, {})
        pre = agent.call("resolve_model", {"model_id": mid})
        resolvable = bool(pre.get("resolvable"))
        t0 = time.perf_counter()
        r = agent.call("run_recipe", {"dataset_path": clean, "timestamp_column": "timestamp",
                                      "target_columns": ["value"],
                                      "recipe": {"estimator": {"model_id": mid}, "fh": fh,
                                                 "eval": {"metrics": ["mae"]}}})
        secs = time.perf_counter() - t0
        err = error_of(r)
        if err:
            if resolvable:
                rows.append((mid, "FAIL", f"resolve_model said resolvable; run failed ({secs:.0f}s)",
                             err))
            else:
                rows.append((mid, "SKIP", f"not runnable here: {pre.get('reason') or 'unresolvable'}",
                             err))
            continue
        score = r.get("backtest_score")
        regime = r.get("training_regime")
        folds = r.get("folds") or (r.get("results") or {}).get("folds")
        declared = card_regime(card)
        notes = [f"{regime}, {folds} fold(s), {secs:.0f}s"]
        status = "PASS"
        if score is None or not math.isfinite(float(score)):
            rows.append((mid, "FAIL", f"backtest_score={score}", None))
            continue
        if declared in ("zero_shot", "zero-shot") and regime != "zero_shot":
            status = "FAIL"
            notes.append(f"card declares zero-shot, run reports {regime}")
        if folds == 1:
            skill = float(score) / (naive_h + 1e-12)
            notes.insert(0, f"MAE {float(score):.4g} = {skill:.2f}x naive")
            if skill > 2:
                status = "FAIL"
            elif skill > 1 and status == "PASS":
                status = "WARN"
        else:
            notes.insert(0, f"MAE {float(score):.4g} over {folds} folds (not comparable to "
                            "a one-holdout score)")
        rows.append((mid, status, "; ".join(notes), None))
        ran.append(mid)
    show(f"run_recipe with every forecasting card (fh=1..{h}, eval=mae)", rows
         or [("-", "SKIP", "no forecasting cards", None)])

    # --------------------------------------------------- gaps and promised horizon
    rows = []
    for mid in ran:
        r = agent.call("run_recipe", {"dataset_path": gapped, "timestamp_column": "timestamp",
                                      "target_columns": ["value"],
                                      "recipe": {"estimator": {"model_id": mid}, "fh": fh,
                                                 "eval": {"metrics": ["mae"]}}})
        err = error_of(r)
        score = r.get("backtest_score")
        if err:
            ok = names_cause(err, "missing") or names_cause(err, "impute") or names_cause(err, "nan")
            rows.append((f"{mid}, real gaps", "PASS" if ok else "FAIL",
                         "refused, names the gaps" if ok else "error does not name the gaps", err))
        elif score is not None and math.isfinite(float(score)):
            rows.append((f"{mid}, real gaps", "PASS", "forecast through the gaps", None))
        else:
            rows.append((f"{mid}, real gaps", "FAIL",
                         f"reported success with backtest_score={score}", None))

        promised = by_id.get(mid, {}).get("prediction_length")
        if promised and int(promised) != h and int(promised) <= len(y) // 4:
            r = agent.call("run_recipe", {"dataset_path": clean, "timestamp_column": "timestamp",
                                          "target_columns": ["value"],
                                          "recipe": {"estimator": {"model_id": mid},
                                                     "fh": list(range(1, int(promised) + 1)),
                                                     "eval": {"metrics": ["mae"]}}})
            err = error_of(r)
            score = r.get("backtest_score")
            if not err and score is not None and math.isfinite(float(score)):
                rows.append((f"{mid}, fh={promised} (card)", "PASS", "ran the card's horizon", None))
            else:
                rows.append((f"{mid}, fh={promised} (card)", "FAIL",
                             "the card's own prediction_length does not run", err))
    show("Gaps and the card's own horizon (models that ran)", rows
         or [("-", "SKIP", "no model ran", None)])

    if failures:
        print(f"{failures} case(s) failed. --show-errors prints what the agent sees.",
              file=sys.stderr)
        return 1
    print("Every model either forecast within the floor or was refused with a named cause. "
          "SKIP is this\nenvironment's limits (packages, weights), not the card; run in the "
          "runtime image to cover them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
