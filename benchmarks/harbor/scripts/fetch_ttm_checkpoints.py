#!/usr/bin/env python
"""Pull TTM checkpoints from the granite series into artifacts/tsfm_models/.

sktime's TTM wrapper requires

    context_length / num_patches == patch_length == patch_stride

and otherwise rewrites the patch geometry, so a zero-shot load fails with
"the model weights in the configuration are mismatched". Only revisions that
satisfy it are fetched.

    # what is published, and the geometry of each
    uv run python benchmarks/harbor/scripts/fetch_ttm_checkpoints.py --list

    # pull one, save it locally, and prove it loads zero-shot
    uv run python benchmarks/harbor/scripts/fetch_ttm_checkpoints.py \\
        --fetch 512-96-r2 --name ttm_512_96

Every fetch is validated before it is written: the invariant is checked, the
checkpoint is reloaded through the same path sktime uses, and a forecast is
produced. A checkpoint that cannot do all three is not left on disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_REPO = "ibm-granite/granite-timeseries-ttm-r2"
DEST_ROOT = Path("artifacts/tsfm_models")


def _ttm_classes():
    """The same import ladder sktime's wrapper walks."""
    try:
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )
        return TinyTimeMixerConfig, TinyTimeMixerForPrediction
    except ImportError:
        from sktime.libs.granite_ttm import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )
        return TinyTimeMixerConfig, TinyTimeMixerForPrediction


def geometry(cfg: dict) -> tuple[bool, str]:
    """Does this config satisfy the invariant sktime enforces?"""
    ctx = cfg.get("context_length")
    npatch = cfg.get("num_patches")
    plen = cfg.get("patch_length")
    pstr = cfg.get("patch_stride")
    if not all(isinstance(v, int) and v > 0 for v in (ctx, npatch, plen, pstr)):
        return False, f"incomplete geometry ctx={ctx} num_patches={npatch} patch_length={plen}"
    size = ctx / npatch
    ok = size == plen == pstr
    detail = (f"ctx={ctx} num_patches={npatch} patch_length={plen} stride={pstr} "
              f"ctx/num_patches={size:g}")
    if not ok:
        detail += f"  -> sktime would rewrite patch_length to {max(1, int(size))}"
    return ok, detail


def cmd_list(repo: str) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    refs = api.list_repo_refs(repo)
    names = [b.name for b in refs.branches]
    print(f"{repo}: {len(names)} revision(s)\n")
    print(f"  {'revision':28} {'ctx':>5} {'horizon':>8}  geometry")
    print("  " + "-" * 76)
    for rev in sorted(names):
        try:
            p = hf_hub_download(repo, "config.json", revision=rev)
            cfg = json.loads(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001 - a listing must not die on one ref
            print(f"  {rev:28} {'-':>5} {'-':>8}  unreadable: {str(exc)[:40]}")
            continue
        ok, detail = geometry(cfg)
        print(f"  {rev:28} {cfg.get('context_length', '-'):>5} "
              f"{cfg.get('prediction_length', '-'):>8}  {'OK  ' if ok else 'SKEW'} {detail}")
    print("\nPick a revision whose geometry says OK; those load zero-shot.")
    return 0


def cmd_fetch(repo: str, revision: str, name: str | None, dest_root: Path) -> int:
    import numpy as np
    import pandas as pd
    from sktime.forecasting.ttm import TinyTimeMixerForecaster

    Config, Model = _ttm_classes()

    print(f"==> {repo}@{revision}")
    cfg_obj = Config.from_pretrained(repo, revision=revision)
    cfg = cfg_obj.to_dict()
    ok, detail = geometry(cfg)
    print(f"    geometry: {detail}")
    if not ok:
        print("    REFUSING: this revision cannot load zero-shot in sktime.",
              file=sys.stderr)
        return 1

    ctx = int(cfg["context_length"])
    horizon = int(cfg["prediction_length"])
    target = dest_root / (name or f"ttm_{ctx}_{horizon}")

    # Stage in a temp dir so a failed validation leaves nothing behind.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "ckpt"
        model = Model.from_pretrained(repo, revision=revision)
        model.save_pretrained(staged)
        print(f"    saved {sum(f.stat().st_size for f in staged.rglob('*') if f.is_file())/1024:.0f} KB")

        # Reload exactly as a card would, and forecast, before anything lands.
        fc = TinyTimeMixerForecaster(model_path=str(staged), fit_strategy="zero-shot")
        n = max(ctx * 2, ctx + horizon + 10)
        t = np.arange(n)
        y = pd.Series(10 + np.sin(t / 7.0) * 2 + t * 0.01)
        fc.fit(y, fh=list(range(1, horizon + 1)))
        pred = np.asarray(fc.predict()).ravel()
        assert len(pred) == horizon, f"expected {horizon} points, got {len(pred)}"
        assert np.isfinite(pred).all(), "non-finite forecast"
        assert pred.std() > 0, "constant forecast: weights did not load"
        print(f"    zero-shot forecast OK, {len(pred)} points, std={pred.std():.4f}")

        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, target)

    print(f"    -> {target}\n")
    print("    card fragment:\n")
    card = {
        "model_id": target.name,
        "model_family": "TinyTimeMixer",
        "sktime_class": "sktime.forecasting.ttm.TinyTimeMixerForecaster",
        "provenance": "pretrained",
        "created_by": "seed",
        "status": "active",
        "source": "local_artifact",
        "hf_repo": None,
        "artifact_path": str(target),
        "model_checkpoint": str(target),
        "params": {"model_path": str(target), "fit_strategy": "zero-shot"},
        "training_regime": "zero_shot",
        "task_ids": ["tsfm_forecasting"],
        "context_length": ctx,
        "prediction_length": horizon,
        "domain": "general",
        "frequency": "any",
        "description": f"TinyTimeMixer from {repo}@{revision}, context {ctx}, horizon {horizon}.",
    }
    print(json.dumps(card, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dest", type=Path, default=DEST_ROOT)
    p.add_argument("--list", action="store_true", help="show revisions and their geometry")
    p.add_argument("--fetch", metavar="REVISION", help="revision to pull")
    p.add_argument("--name", help="directory name under --dest (default ttm_<ctx>_<horizon>)")
    args = p.parse_args()

    if args.list:
        return cmd_list(args.repo)
    if args.fetch:
        return cmd_fetch(args.repo, args.fetch, args.name, args.dest)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
