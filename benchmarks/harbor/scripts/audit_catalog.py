#!/usr/bin/env python
"""Flag catalog cards whose defaults do something the card does not say.

An empty `params` is not neutral. It hands the wrapper its own defaults, and
those defaults both name a checkpoint and pick a fit strategy:

    TinyTimeMixerForecaster   model_path="ibm/TTM"   fit_strategy="minimal"
    TSPulseAnomalyDetector    model_path="ibm-granite/granite-timeseries-tspulse-r1"
    TSPulseClassifier         model_path=...  revision="tspulse-block-dualhead-512-p16-r1"

So a card with `"params": {}` downloads weights at run time that
`preload_models.py` never sees, because there is nothing in the card to
collect. It passes every check on a connected machine and fails inside the
image. The same emptiness leaves TTM on "minimal", which fine-tunes.

    uv run python benchmarks/harbor/scripts/audit_catalog.py
    AOB_MODEL_CATALOG=... uv run python benchmarks/harbor/scripts/audit_catalog.py

Reports only; it never edits. Exit 1 when any card is flagged.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")
PATH_PARAMS = ("model_path", "checkpoint_path", "repo_id",
               "pretrained_model_name_or_path", "tokenizer_path")
SWITCHES = ("fit_strategy", "train_model")


def catalog_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    for env in ("AOB_MODEL_CATALOG",):
        if os.environ.get(env):
            return Path(os.environ[env])
    if os.environ.get("SCENARIOS_DATA_DIR"):
        return Path(os.environ["SCENARIOS_DATA_DIR"]) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def signature(card: dict):
    dotted = card.get("sktime_class")
    if not dotted or "." not in dotted:
        return None
    mod, _, name = dotted.rpartition(".")
    try:
        return inspect.signature(getattr(importlib.import_module(mod), name).__init__).parameters
    except Exception:  # noqa: BLE001 - an uninstallable wrapper is not this tool's problem
        return None


def main() -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    args = p.parse_args()

    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    raw = json.loads(cat.read_text(encoding="utf-8"))
    cards = raw if isinstance(raw, list) else raw.get("docs", [raw])
    active = [c for c in cards if (c.get("status") or "active") == "active"]

    sys.path.insert(0, "src")
    print(f"{cat}\n{len(cards)} card(s), {len(active)} active\n")
    findings = 0

    print("A. active cards that fetch weights at run time, invisible to preload")
    for c in active:
        params = c.get("params") or {}
        if c.get("hf_repo") or any(params.get(k) for k in PATH_PARAMS):
            continue
        sig = signature(c)
        if not sig:
            continue
        for key in ("model_path", "checkpoint_path", "pretrained_model_name_or_path"):
            if key in sig and sig[key].default not in (inspect._empty, None):
                print(f"   {c.get('model_id', '?'):22} pulls {sig[key].default!r} "
                      f"via the wrapper default; set params.{key} and hf_repo")
                findings += 1
                break

    print("\nB. active forecasting cards that train on fit")
    for c in active:
        if not any("forecast" in str(t) for t in (c.get("task_ids") or [])):
            continue
        if c.get("training_regime") == "fine_tune":
            continue           # declared, and reported as TUNED by the smoke test
        sig, params = signature(c), (c.get("params") or {})
        if not sig:
            continue
        for key in SWITCHES:
            if key in sig and key not in params:
                print(f"   {c.get('model_id', '?'):22} no params.{key}; wrapper default "
                      f"is {sig[key].default!r}")
                findings += 1

    print("\nC. active cards resolving to the same weights")
    seen: dict[str, list[str]] = {}
    for c in active:
        params = c.get("params") or {}
        ref = str(c.get("hf_repo") or params.get("model_path")
                  or params.get("checkpoint_path") or "")
        if ref:
            seen.setdefault(ref, []).append(c.get("model_id", "?"))
    for ref, ids in seen.items():
        if len(ids) > 1:
            print(f"   {ref}  <- {', '.join(ids)}")
            findings += 1

    print("\nD. declared geometry that cannot be right")
    for c in active:
        ctx = c.get("context_length")
        if isinstance(ctx, int) and ctx < 8:
            print(f"   {c.get('model_id', '?'):22} context_length={ctx}")
            findings += 1

    print()
    if findings:
        print(f"{findings} finding(s).", file=sys.stderr)
        return 1
    print("nothing flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
