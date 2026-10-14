#!/usr/bin/env python
"""Use the feature catalog the way an agent does: through the TSFM MCP tools.

`smoke_features.py` proves the extractors and transforms work when called
directly. An agent never calls them directly. It reads `list_features`,
`recipe_template` and the tool docstrings, then sends JSON to `extract_features`,
`select_features`, `run_tabular_recipe` and `run_recipe`. This script sends the
JSON an agent would send and checks each answer against one contract:

    the tool either does what was asked, or returns an error that names the cause.

A success that silently did something else (a feature dropped, a transform
ignored, a gap turned into 0.0) is a FAIL, because the agent cannot tell it apart
from a real answer and will build on it.

Calls go through FastMCP's `call_tool`, the same argument validation and JSON
serialisation the stdio transport uses. The store is the in-memory one seeded
from the catalog you pass, so no CouchDB is needed and nothing is written to one.

    uv run python benchmarks/harbor/scripts/smoke_agent_features.py --catalog <feature_catalog.json>

Exit 1 when any FAIL is reported, so it works as a gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

DEFAULT_SERIES = Path("src/couchdb/scenarios_data/shared/iot/chiller_6.json")


class Agent:
    """Talks to the TSFM server through its MCP tool boundary, as an agent would."""

    def __init__(self, main):
        self.mcp = main.mcp

    def call(self, tool: str, args: dict) -> dict:
        content, _ = asyncio.run(self.mcp.call_tool(tool, args))
        payload = json.loads(content[0].text) if content else {}
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        return payload


def error_of(r) -> str | None:
    return r.get("error") if isinstance(r, dict) else None


def names_cause(err: str | None, *words: str) -> bool:
    return bool(err) and all(w.lower() in err.lower() for w in words)


def seed(main, cards):
    """Load the catalog into both stores the tools read (main._STORE, main._FEATURE_STORE)."""
    from servers.tsfm.stores import feature_store as fs

    for store in {id(main._STORE): main._STORE,
                  id(main._FEATURE_STORE): main._FEATURE_STORE}.values():
        for c in cards:
            doc = dict(c)
            doc.setdefault("_id", f"feature:{doc['feature_id']}")
            store.put(fs.collection_name(), doc)


def datasets(series_path: Path):
    """File pointers an agent would be handed: a real column with its own gaps, the same
    column filled, and a small instances-by-time table for the tabular tools."""
    import numpy as np
    import pandas as pd

    from servers.tsfm.io import refs

    records = json.loads(series_path.read_text(encoding="utf-8"))
    skip = {"asset_id", "timestamp"}
    best = None
    for k, v in records[0].items():
        if k in skip or not isinstance(v, (int, float)):
            continue
        arr = np.array([np.nan if r.get(k) is None else r.get(k) for r in records], float)
        n_gap = int(np.isnan(arr).sum())
        if 0 < n_gap <= 0.05 * len(arr) and np.nanstd(arr) > 0 and (arr == 0).mean() < 0.2:
            cv = np.nanstd(arr) / (abs(np.nanmean(arr)) + 1e-9)
            if best is None or cv > best[0]:
                best = (cv, k, arr)
    if best is None:
        raise SystemExit(f"no numeric column with a few gaps in {series_path}")
    _, column, raw = best
    idx = np.arange(len(raw))
    good = ~np.isnan(raw)
    filled = np.interp(idx, idx[good], raw[good])
    gapped_ref = refs.materialize_iot(raw, asset_id="agent_smoke_gapped")
    clean_ref = refs.materialize_iot(filled, asset_id="agent_smoke_clean")

    # Instances for classification: windows of the real column, labelled by whether
    # the window sits above the series median. Learnable, so a run that succeeds with
    # the features it was asked for should score well above chance.
    w = 48
    starts = range(0, len(filled) - w, w)
    rows = [filled[s:s + w] for s in starts]
    med = float(np.median(filled))
    labels = [int(r.mean() > med) for r in rows]
    table = pd.DataFrame(rows, columns=[f"t{i}" for i in range(w)])
    table["label"] = labels
    os.makedirs(refs.WORKDIR, exist_ok=True)
    tab = os.path.join(refs.WORKDIR, "agent_smoke_table.csv")
    table.to_csv(tab, index=False)
    gapped_table = table.copy()
    gapped_table.iloc[::4, 7] = np.nan
    tab_gap = os.path.join(refs.WORKDIR, "agent_smoke_table_gapped.csv")
    gapped_table.to_csv(tab_gap, index=False)
    return {
        "column": column, "n_gaps": int((~good).sum()), "n": len(raw),
        "clean": clean_ref, "gapped": gapped_ref,
        "table": f"file://{tab}", "table_gapped": f"file://{tab_gap}",
        "n_instances": len(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, required=True,
                   help="path to the feature_catalog.json to serve to the agent")
    p.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    p.add_argument("--show-errors", action="store_true",
                   help="print the full error text the agent receives for every case")
    args = p.parse_args()

    if not args.catalog.is_file():
        print(f"no catalog at {args.catalog}", file=sys.stderr)
        return 1
    if not args.series.is_file():
        print(f"no series at {args.series}", file=sys.stderr)
        return 1

    os.environ["TSFM_STORE"] = "memory"   # before the server module builds its stores
    sys.path.insert(0, "src")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import logging

    logging.disable(logging.CRITICAL)
    # sktime and sklearn warn on every fit, and the server code resets warning filters,
    # so drop the display itself. The table is the output; warnings would bury it.
    warnings.showwarning = lambda *a, **k: None

    from smoke_features import code_identity

    from servers.tsfm import main as server
    from servers.tsfm.reasoning import feature_selection as FS

    raw = json.loads(args.catalog.read_text(encoding="utf-8"))
    cards = [c for c in (raw if isinstance(raw, list) else raw.get("docs", [raw]))
             if (c.get("status") or "active") == "active"]
    seed(server, cards)
    agent = Agent(server)
    data = datasets(args.series)
    extractor_cards = [c for c in cards if c.get("kind") == "extractor"]
    transform_cards = [c for c in cards if c.get("kind", "transform") == "transform"]

    print(f"catalog  : {args.catalog}  ({len(extractor_cards)} extractor, "
          f"{len(transform_cards)} transform cards)")
    print(f"series   : {args.series.name}  column={data['column']!r}  n={data['n']}  "
          f"({data['n_gaps']} real gaps)")
    print(f"code     : {code_identity()}")
    print("transport: FastMCP call_tool, in-memory store seeded from --catalog")
    print()

    failures = 0

    def show(section, rows):
        nonlocal failures
        print(section)
        print("  " + "-" * 78)
        for name, status, note, err in rows:
            print(f"  {name[:42]:42} {status:5}  {note}")
            if args.show_errors and err:
                print(f"  {'':42}        agent sees: {' '.join(err.split())[:200]}")
            if status == "FAIL":
                failures += 1
        print()

    # ------------------------------------------------------------------ discovery
    rows = []
    r = agent.call("count_features", {})
    want = (len(extractor_cards), len(transform_cards))
    got = (r.get("extractors"), r.get("transforms"))
    rows.append(("count_features", "PASS" if got == want else "FAIL",
                 f"extractors={got[0]} transforms={got[1]}"
                 + ("" if got == want else f"; catalog has {want[0]} / {want[1]}"), None))

    r = agent.call("list_features", {"kind": "extractor"})
    listed = [f.get("extractor_name") or f.get("feature_id")
              for f in (r.get("features") or [])]
    listed_ids = [f.get("feature_id") for f in (r.get("features") or []) if f.get("feature_id")]
    r2 = agent.call("extract_features", {"dataset_path": data["clean"], "extractors": listed,
                                         "target_columns": ["value"]})
    err = error_of(r2)
    rows.append(("list_features -> extract_features",
                 "PASS" if listed and not err else "FAIL",
                 (f"all {len(listed)} listed extractors accepted" if not err
                  else f"{len(listed)} listed; extract_features refused them"), err))

    r = agent.call("search_features", {"text": "entropy"})
    hits = [f.get("extractor_name") for f in (r.get("features") or [])
            if f.get("kind") == "extractor" and f.get("extractor_name")]
    if hits:
        r2 = agent.call("extract_features", {"dataset_path": data["clean"], "extractors": hits,
                                             "target_columns": ["value"]})
        err = error_of(r2)
        rows.append(("search_features('entropy') -> extract",
                     "FAIL" if err else "PASS",
                     f"{len(hits)} hit(s){'; refused' if err else ', all usable'}", err))
    else:
        rows.append(("search_features('entropy')", "WARN",
                     "no extractor hits; the catalog has no descriptions mentioning entropy",
                     error_of(r)))

    sample = listed_ids[:3]
    r = agent.call("describe_features", {"names": sample})
    found = [f.get("feature_id") for f in (r.get("features") or [])]
    rows.append(("describe_features(3 listed names)",
                 "PASS" if len(found) == len(sample) and not r.get("unknown") else "FAIL",
                 f"found {len(found)}/{len(sample)}, unknown={r.get('unknown')}", error_of(r)))

    r = agent.call("recipe_template", {})
    blocks = " ".join(r.get("optional_blocks") or [])
    documented = all(s in blocks for s in ("extractors", "sktime_class"))
    rows.append(("recipe_template: transform shapes",
                 "PASS" if documented else "WARN",
                 "documents the transform spec shapes" if documented
                 else "says 'list of transform specs' but not which shapes are accepted",
                 None))
    show("Discovery (what the agent reads before acting)", rows)

    # ------------------------------------------------------------ extract / select
    rows = []
    five = [n for n in ("mean", "std", "slope", "autocorr1", "spectral_entropy")
            if n in FS.EXTRACTORS]
    r = agent.call("extract_features", {"dataset_path": data["clean"], "extractors": five,
                                        "target_columns": ["value"]})
    vals = (r.get("features") or [[]])[0]
    ok = not error_of(r) and len(vals) == len(five) and all(v is not None for v in vals)
    rows.append(("extract_features, clean", "PASS" if ok else "FAIL",
                 f"{len(vals)} values for {len(five)} extractors", error_of(r)))

    r = agent.call("extract_features", {"dataset_path": data["gapped"], "extractors": five,
                                        "target_columns": ["value"]})
    err = error_of(r)
    rows.append(("extract_features, real gaps", "PASS" if names_cause(err, "missing", "impute")
                 else "FAIL",
                 "refused, names impute" if err else
                 f"returned {r.get('features')} from a series with {data['n_gaps']} gaps", err))

    r = agent.call("extract_features", {"dataset_path": data["gapped"], "extractors": five,
                                        "target_columns": ["value"], "impute": "interpolate"})
    vals = (r.get("features") or [[]])[0]
    ok = not error_of(r) and all(v is not None for v in vals)
    rows.append(("extract_features, impute=interpolate", "PASS" if ok else "FAIL",
                 f"{len(vals)} values" if ok else "failed", error_of(r)))

    r = agent.call("extract_features", {"dataset_path": data["clean"],
                                        "extractors": ["mean", "not_an_extractor"],
                                        "target_columns": ["value"]})
    err = error_of(r)
    rows.append(("extract_features, unknown name",
                 "PASS" if names_cause(err, "not_an_extractor") else "FAIL",
                 "refused, names it" if err else "accepted an unknown name", err))

    r = agent.call("select_features", {"dataset_path": data["clean"], "channel": "value",
                                       "extractors": five})
    rows.append(("select_features, clean", "FAIL" if error_of(r) else "PASS",
                 f"selected {r.get('selected')}" if not error_of(r) else "failed", error_of(r)))

    r = agent.call("select_features", {"dataset_path": data["gapped"], "channel": "value",
                                       "extractors": five})
    err = error_of(r)
    rows.append(("select_features, real gaps",
                 "PASS" if names_cause(err, "missing", "impute") else "FAIL",
                 "refused, names impute" if err else "ranked a gapped series", err))
    show("Extract and select", rows)

    # ---------------------------------------------------------------- tabular recipe
    rows = []
    base = {"task": "tsfm_classification",
            "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                          "params": {"n_estimators": 50, "random_state": 0}}}

    def tabular(transforms, table="table"):
        recipe = dict(base)
        if transforms is not None:
            recipe["transforms"] = transforms
        return agent.call("run_tabular_recipe", {"dataset_path": data[table],
                                                 "recipe": recipe, "label_column": "label"})

    r = tabular([{"extractors": ["mean", "std"]}])
    ok = not error_of(r) and r.get("n_features") == 2
    rows.append(("extractors=[mean, std]", "PASS" if ok else "FAIL",
                 f"n_features={r.get('n_features')} cv={r.get('cv_score')}", error_of(r)))

    r = tabular([{"extractors": ["mean", "not_an_extractor"]}])
    err = error_of(r)
    if names_cause(err, "not_an_extractor"):
        rows.append(("extractors=[mean, not_an_extractor]", "PASS", "refused, names it", err))
    else:
        rows.append(("extractors=[mean, not_an_extractor]", "FAIL",
                     (f"succeeded with n_features={r.get('n_features')}; the unknown name "
                      "was dropped without a word") if not err
                     else "error does not name the unknown extractor", err))

    r = tabular([{"extractors": ["not_an_extractor"]}])
    err = error_of(r)
    rows.append(("extractors=[not_an_extractor]",
                 "PASS" if names_cause(err, "not_an_extractor") else "FAIL",
                 "refused, names it" if names_cause(err, "not_an_extractor")
                 else ("error does not name the unknown extractor" if err
                       else f"succeeded with n_features={r.get('n_features')}"), err))

    full = len(FS.EXTRACTORS)
    for card in transform_cards:
        fid = card["feature_id"]
        r = tabular([{"feature_id": fid}])
        err = error_of(r)
        if err:
            status = "PASS" if fid in err else "FAIL"
            note = "refused, names the card" if status == "PASS" else "error does not name the card"
        elif r.get("n_features") == full:
            status, note = "FAIL", (f"ignored: ran the full {full}-extractor library "
                                    "instead and reported success")
        else:
            status, note = "PASS", f"applied, n_features={r.get('n_features')}"
        rows.append((f"transforms=[{{feature_id: {fid}}}]"[:42], status, note, err))

    r = tabular([{"flops_select": True}])
    rows.append(("flops_select", "FAIL" if error_of(r) else "PASS",
                 f"n_features={r.get('n_features')}" if not error_of(r) else "failed",
                 error_of(r)))

    r = tabular([{"extractors": ["mean", "std"]}], table="table_gapped")
    err = error_of(r)
    rows.append(("gapped instances, extractors=[mean, std]",
                 "PASS" if names_cause(err, "missing") else "FAIL",
                 "refused, names the gaps" if names_cause(err, "missing")
                 else (f"succeeded (cv={r.get('cv_score')}); gapped rows were scored, "
                       "their NaN features became 0.0") if not err
                 else "error does not name the gaps", err))
    show(f"run_tabular_recipe ({data['n_instances']} instances, classification)", rows)

    # ------------------------------------------------------------------- run_recipe
    rows = []
    naive = {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
             "params": {"strategy": "last"}}

    def forecast(recipe, ref="clean"):
        return agent.call("run_recipe", {"dataset_path": data[ref],
                                         "timestamp_column": "timestamp",
                                         "target_columns": ["value"], "recipe": recipe})

    r = forecast({"estimator": naive, "fh": [1, 2, 3]})
    rows.append(("no transforms (baseline)", "FAIL" if error_of(r) else "PASS",
                 "ran" if not error_of(r) else "failed", error_of(r)))

    r = forecast({"estimator": naive, "fh": [1, 2, 3],
                  "transforms": [{"sktime_class":
                                  "sktime.transformations.series.exponent.ExponentTransformer",
                                  "params": {"power": 1.0}}]})
    rows.append(("transforms=[{sktime_class}]", "FAIL" if error_of(r) else "PASS",
                 "ran" if not error_of(r) else "failed", error_of(r)))

    for card in transform_cards:
        fid = card["feature_id"]
        r = forecast({"estimator": naive, "fh": [1, 2, 3],
                      "transforms": [{"feature_id": fid}]})
        err = error_of(r)
        if not err:
            status, note = "PASS", "applied"
        elif fid in err:
            status, note = "PASS", "refused, names the card"
        else:
            status, note = "FAIL", ("offered by list_features, unusable here; error does not "
                                    "name the card")
        rows.append((f"transforms=[{{feature_id: {fid}}}]"[:42], status, note, err))

    r = forecast({"estimator": naive, "fh": [1, 2, 3]}, ref="gapped")
    err = error_of(r)
    rows.append(("real gaps, no impute",
                 "PASS" if names_cause(err, "impute") else "FAIL",
                 "refused, names impute" if names_cause(err, "impute")
                 else ("ran on a gapped series" if not err else "error does not name impute"),
                 err))
    show("run_recipe (forecasting)", rows)

    if failures:
        print(f"{failures} case(s) failed: the tool returned something the agent would take "
              "as an answer,\nor an error that does not say what to change. "
              "--show-errors prints what the agent sees.", file=sys.stderr)
        return 1
    print("Every case either did what was asked or returned an error that names the cause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
