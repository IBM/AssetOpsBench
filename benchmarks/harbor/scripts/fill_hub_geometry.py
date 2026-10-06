#!/usr/bin/env python
"""Read each Hub card's real geometry from its config.json and record it.

A card that declares no context_length/prediction_length is not neutral. The
wrapper fills the gap from the forecasting horizon and its own module default,
then overrides the checkpoint's config with the result. PatchTSMixerForecaster
does exactly that:

    merged = {**hub_cfg.to_dict(), **self._build_model_config(ctx, pred, n_ch)}
    ...
    PatchTSMixerForForecast.from_pretrained(..., config=config,
                                            ignore_mismatched_sizes=True)

Two consequences worth naming. The resolved lengths win over the checkpoint's
own, so an undeclared card silently reshapes the model. And
ignore_mismatched_sizes=True means any layer that still does not line up is
dropped and randomly re-initialised rather than raising, so a card that serves
without training can return numbers from a partly random head.

    uv run python benchmarks/harbor/scripts/fill_hub_geometry.py
    uv run python benchmarks/harbor/scripts/fill_hub_geometry.py --write

Fills only fields the card leaves null. A declared value that disagrees with
the checkpoint is reported, never overwritten: that disagreement is a decision,
not a typo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")

# Architectures nest their geometry differently. Chronos keeps it under
# chronos_config; the IBM family keeps it at the top level; some report window
# sizes under seq_len/pred_len. Try each rather than assuming one shape.
CTX_KEYS = ("context_length", "seq_len", "max_context_length", "input_size")
HORIZON_KEYS = ("prediction_length", "pred_len", "forecast_horizon", "horizon")


def dig(cfg: dict, keys: tuple[str, ...]) -> int | None:
    for scope in (cfg, cfg.get("chronos_config") or {}, cfg.get("model_config") or {}):
        if not isinstance(scope, dict):
            continue
        for k in keys:
            v = scope.get(k)
            if isinstance(v, int) and v > 0:
                return v
    return None


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


def fetch(repo: str, revision: str) -> dict | None:
    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(repo, "config.json", revision=revision)
    except Exception as exc:  # noqa: BLE001 - one unreachable repo must not stop the sweep
        print(f"  {repo:52} unreadable: {str(exc).splitlines()[0][:40]}", file=sys.stderr)
        return None
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--write", action="store_true", help="fill null fields in place")
    args = p.parse_args()

    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    raw = json.loads(cat.read_text(encoding="utf-8"))
    cards = raw if isinstance(raw, list) else raw.get("docs", [raw])

    print(f"catalog: {cat}\n")
    print(f"  {'model':44} {'ctx':>6} {'horizon':>8} {'ch':>4}  status")
    print("  " + "-" * 82)

    filled = conflicts = multich = 0
    for c in cards:
        repo = c.get("hf_repo")
        if not repo:
            continue
        repo_id, _, rev = str(repo).partition("@")
        cfg = fetch(repo_id, rev or "main")
        if cfg is None:
            continue

        ctx, horizon = dig(cfg, CTX_KEYS), dig(cfg, HORIZON_KEYS)
        channels = dig(cfg, ("num_input_channels",)) or 1
        notes = []

        for field, found in (("context_length", ctx), ("prediction_length", horizon)):
            have = c.get(field)
            if found is None:
                notes.append(f"{field} not in config.json")
            elif have is None:
                c[field] = found
                filled += 1
                notes.append(f"filled {field}={found}")
            elif have != found:
                conflicts += 1
                notes.append(f"CONFLICT {field}: card={have} checkpoint={found}")

        if channels > 1:
            multich += 1
            notes.append(f"{channels} input channels; the tsfm engine calls "
                         "fit(y, fh) univariate, so the wrapper rewrites this to 1 "
                         "and the mismatched weights are re-initialised")

        print(f"  {c.get('model_id', '?'):44} {ctx if ctx else '-':>6} "
              f"{horizon if horizon else '-':>8} {channels:>4}  {'; '.join(notes)}")

    print()
    if args.write and filled:
        cat.write_text(json.dumps(cards, indent=2) + "\n", encoding="utf-8")
        print(f"filled {filled} field(s) in {cat}")
    elif filled:
        print(f"{filled} field(s) would be filled; pass --write")
    if conflicts:
        print(f"{conflicts} declared value(s) disagree with the checkpoint. Left alone: "
              "decide which is right.", file=sys.stderr)
    if multich:
        print(f"{multich} card(s) are multivariate checkpoints driven univariately. "
              "Their weights\nare partly re-initialised at load, so a zero-shot score "
              "from them is not a score.", file=sys.stderr)
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
