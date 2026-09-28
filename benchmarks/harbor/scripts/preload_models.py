#!/usr/bin/env python
"""Report, warm and verify the HuggingFace cache an AssetOpsBench catalog needs.

Three modes over one source of truth, the model catalog:

    preload_models.py --report     # ask the Hub how big each repo is
    preload_models.py --download   # fill the cache (resumable)
    preload_models.py --check      # offline: does the cache satisfy every card?

`--report` and `--check` need only `huggingface_hub`. None of the modes needs
torch, sktime or any per-model library, so the cache can be built and validated
before any of the dependency work lands.

Where the cache goes
--------------------
Whatever `huggingface_hub` resolves, which is `HF_HUB_CACHE` if set, else
`$HF_HOME/hub`, else `~/.cache/huggingface/hub`. Every mode prints the path it
is using, so set the variable and read it back rather than trusting a default.

A warmed cache is safe to mount read-only into a trial: with the files present
and `HF_HUB_OFFLINE=1`, resolution reads the tree and writes nothing, and a repo
that is missing raises `LocalEntryNotFoundError` instead of quietly going to the
network. `--check` is that same code path, run ahead of time.

Which field is authoritative
----------------------------
`params.model_path`, not `hf_repo`. The resolver builds the estimator with
`Est(**card["params"])`, so `model_path` is the string that reaches
`from_pretrained`. `hf_repo` is metadata and can drift; it is used only as a
fallback, and a disagreement between the two is reported because it means one
of them is wrong.

Fine-tuned cards are skipped: `register_finetuned` points `model_path` at a
local checkpoint directory, and there is nothing on the Hub to fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# The catalog is one global file, not per-scenario data, and the copy in this
# repo is only an example. Resolution order, most explicit first:
#
#   1. --catalog
#   2. $AOB_MODEL_CATALOG
#   3. $SCENARIOS_DATA_DIR/shared/tsfm/model_catalog.json
#   4. the in-repo example
#
# Step 3 is the one that matters. At run time CouchDB is seeded from the
# manifest key "model_catalog": "shared/tsfm/model_catalog.json", resolved
# against SCENARIOS_DATA_DIR. Defaulting to the same file means the weights
# baked into an image are the weights the agent can actually discover. Point
# them at different files and every model outside the example misses the cache.
CATALOG_REL = Path("shared/tsfm/model_catalog.json")
EXAMPLE_CATALOG = Path("src/couchdb/scenarios_data") / CATALOG_REL


def resolve_catalog(explicit: Path | None) -> tuple[Path, str]:
    """Return (path, where_it_came_from)."""
    if explicit is not None:
        return explicit, "--catalog"
    env = os.environ.get("AOB_MODEL_CATALOG")
    if env:
        return Path(env), "$AOB_MODEL_CATALOG"
    root = os.environ.get("SCENARIOS_DATA_DIR")
    if root:
        return Path(root) / CATALOG_REL, "$SCENARIOS_DATA_DIR"
    return EXAMPLE_CATALOG, "in-repo EXAMPLE"

# "owner/name". A local checkpoint is an absolute path, or a relative one that
# exists on disk. Conservative on purpose: a false positive costs a failed
# download, a false negative costs a missing weight at run time.
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_ZERO_SHOT = {"zero-shot", "zero_shot", "zeroshot"}


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
def load_cards(path: Path) -> list[dict]:
    """Accept a bare list, a {"docs": [...]} wrapper, or a single card."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("docs", raw.get("rows", [raw]))
    if isinstance(raw, dict):
        raw = [raw]
    return [c for c in raw if isinstance(c, dict)]


def status_of(card: dict) -> str:
    """A card with no status is treated as active, matching model_store.list_models."""
    return str(card.get("status") or "active")


def classify(card: dict) -> tuple[str, str | None, str]:
    """Return (kind, target, why). kind is 'hub', 'local', 'runtime' or 'none'.

    Branch on what the card DECLARES, not on the shape of the path. Path shape
    cannot tell a two-segment local checkpoint ("tuned/ttm_ft") from a Hub repo
    id, and guessing wrong sends a local model to huggingface.co.

      runtime  created_by starts with "agent." - the checkpoint does not exist
               until a trial writes it, so it must not be verified at build time.
               provenance alone is NOT the test: a seeded card can legitimately
               be provenance="finetuned" when you ship a fine-tuned checkpoint.
      hub      hf_repo is set - weights come from the Hub.
      local    source == "local_artifact" with no hf_repo - a directory on disk.
      none     no model_path at all, e.g. a classical forecaster that fits from
               scratch.
    """
    model_id = card.get("model_id") or "<unnamed>"
    path = (card.get("params") or {}).get("model_path")
    created_by = str(card.get("created_by") or "")
    hf_repo = card.get("hf_repo")
    source = card.get("source")

    if created_by.startswith("agent."):
        return "runtime", None, f"{model_id}: written at run time by {created_by}"
    if not path:
        if hf_repo:
            rev0 = (card.get("params") or {}).get("revision")
            tgt = f"{hf_repo}@{rev0}" if rev0 else str(hf_repo)
            return "hub", tgt, f"{model_id}: {tgt} (via hf_repo; no params.model_path)"
        return "none", None, f"{model_id}: no params.model_path (classical model?)"

    path = str(path)
    rev = (card.get("params") or {}).get("revision")
    suffix = f"@{rev}" if rev else ""
    if hf_repo:
        if str(hf_repo) != path:
            return "hub", path + suffix, (
                f"{model_id}: WARNING params.model_path={path} disagrees with "
                f"hf_repo={hf_repo}; loading follows model_path"
            )
        return "hub", path + suffix, f"{model_id}: {path}{suffix}"

    if source == "local_artifact":
        return "local", path, f"{model_id}: local checkpoint {path}"

    # Nothing declared. Fall back to shape, and say so, because this is the
    # case that silently sends a local path to the Hub.
    if _REPO_RE.match(path):
        return "hub", path + suffix, (
            f"{model_id}: {path}{suffix} (GUESSED from path shape; set hf_repo or "
            f'source="local_artifact" to make this explicit)'
        )
    return "local", path, (
        f"{model_id}: local checkpoint {path} (guessed; no source declared)"
    )


def check_local(target: str, root: Path) -> tuple[bool, str]:
    """A local card is satisfied when its directory exists and holds a config."""
    p = Path(target)
    if not p.is_absolute():
        p = root / p
    if not p.is_dir():
        return False, f"missing directory {p}"
    if not (p / "config.json").exists():
        return False, f"{p} has no config.json (not a save_pretrained checkpoint)"
    return True, str(p)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def human(n: float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def tree_size(path: Path) -> int:
    """Bytes actually on disk. Follows the blob symlinks a snapshot uses, and
    counts each blob once so a repo is not double counted."""
    seen: set[int] = set()
    total = 0
    for p in path.rglob("*"):
        try:
            st = p.stat()
        except OSError:
            continue
        if not p.is_file() or st.st_ino in seen:
            continue
        seen.add(st.st_ino)
        total += st.st_size
    return total


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def split_ref(ref: str, override: str | None) -> tuple[str, str | None]:
    """"org/name@branch" -> ("org/name", "branch"). --revision overrides the card."""
    if override:
        return ref.split("@", 1)[0], override
    if "@" in ref:
        repo, rev = ref.split("@", 1)
        return repo, rev
    return ref, None


def report(repos: list[str], revision: str | None) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    total = 0
    unknown = 0
    print(f"{'repo':52} {'files':>6} {'size':>10}")
    print("-" * 74)
    for ref in repos:
        repo, rev = split_ref(ref, revision)
        try:
            info = api.model_info(repo, revision=rev, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - a report must not die on one repo
            print(f"{ref:52} {'-':>6} {'ERROR':>10}  {exc}")
            continue
        sizes = [s.size for s in (info.siblings or [])]
        unknown += sum(1 for s in sizes if s is None)
        known = sum(s for s in sizes if s)
        total += known
        print(f"{ref:52} {len(sizes):>6} {human(known):>10}  sha={(info.sha or '')[:8]}")
    print("-" * 74)
    print(f"{'TOTAL':52} {'':>6} {human(total):>10}")
    if unknown:
        print(
            f"\n{unknown} file(s) reported no size, so the total is a floor, not a ceiling."
        )
    free = shutil.disk_usage(cache_dir().parent if cache_dir().exists() else Path.home()).free
    print(f"\ncache dir : {cache_dir()}")
    print(f"free disk : {human(free)}")
    if free < total * 1.1:
        print("WARNING: less free space than the download needs.")
    return 0


def download(repos: list[str], revision: str | None, workers: int) -> int:
    from huggingface_hub import snapshot_download

    dest = cache_dir()
    print(f"cache dir : {dest}")
    print(f"workers   : {workers} (per repo)\n")

    failed: list[tuple[str, str]] = []
    grand = 0
    for n, ref in enumerate(repos, 1):
        repo, rev = split_ref(ref, revision)
        print(f"[{n}/{len(repos)}] {ref}")
        started = time.monotonic()
        try:
            where = Path(snapshot_download(repo, revision=rev, max_workers=workers))
        except Exception as exc:  # noqa: BLE001
            print(f"          FAILED: {exc}", file=sys.stderr)
            failed.append((ref, str(exc)))
            continue
        # The repo root is two levels up from snapshots/<sha>.
        size = tree_size(where.parent.parent)
        grand += size
        print(f"          {human(size)} in {time.monotonic() - started:.0f}s")

    print(f"\n{len(repos) - len(failed)}/{len(repos)} repos cached, {human(grand)} on disk")
    if failed:
        print(f"\n{len(failed)} failed:", file=sys.stderr)
        for repo, err in failed:
            print(f"  {repo}: {err}", file=sys.stderr)
        print("\nRe-run to retry; completed repos are skipped and partial files resume.",
              file=sys.stderr)
        return 1
    print("\nNext: verify it offline before you depend on it:\n"
          f"  HF_HUB_CACHE={dest} HF_HUB_OFFLINE=1 {sys.argv[0]} --check")
    return 0


def validate_cards(cards: list[dict]) -> int:
    """Check the cards themselves, not just where their weights live.

    --check verifies that a checkpoint is on disk or in the cache. That is
    necessary and not sufficient: a card can point at real weights and still be
    wrong. The two failures worth catching before a run:

      schema      the repo's own validator, so a bad card fails here rather
                  than at seed time. Notably it requires base_model_id on a
                  finetuned card.
      serve pins  params.fit_strategy and training_regime. Without both, TTM's
                  default fit_strategy="minimal" re-tunes the weights on every
                  fit and run_recipe takes the expanding-window refit path.
                  This one is silent: the model loads, forecasts, and is wrong.
    """
    problems: list[str] = []

    try:
        sys.path.insert(0, "src")
        from servers.tsfm.core import schemas
    except ImportError:
        schemas = None
        print("  schema validator unavailable (run from the repo root to enable)")

    for c in cards:
        mid = c.get("model_id", "<unnamed>")
        if schemas is not None:
            try:
                schemas.validate_model(dict(c))
            except Exception as exc:  # noqa: BLE001 - report every card, not the first
                problems.append(f"{mid}: schema: {str(exc).splitlines()[0][:120]}")

        params = c.get("params") or {}
        if str(params.get("fit_strategy", "")).lower() not in _ZERO_SHOT:
            problems.append(
                f"{mid}: params.fit_strategy is {params.get('fit_strategy')!r}; "
                'a checkpoint card must pin "zero-shot" or it re-tunes on every fit')
        if c.get("training_regime") != "zero_shot":
            problems.append(
                f"{mid}: training_regime is {c.get('training_regime')!r}; "
                'pin "zero_shot" or run_recipe takes the refit path')

    if problems:
        print(f"\n{len(problems)} card problem(s):", file=sys.stderr)
        for p_ in problems:
            print(f"  {p_}", file=sys.stderr)
        return 1
    print(f"  {len(cards)} card(s) validate, and every one pins zero-shot serving")
    return 0


def check(repos: list[str], revision: str | None) -> int:
    """Resolve every repo with the network disabled, exactly as a trial will."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    print(f"cache dir : {cache_dir()}")
    print("offline   : HF_HUB_OFFLINE=1\n")

    missing: list[str] = []
    for ref in repos:
        repo, rev = split_ref(ref, revision)
        try:
            where = Path(snapshot_download(repo, revision=rev))
            print(f"  OK      {ref:52} {human(tree_size(where.parent.parent)):>10}")
        except Exception as exc:  # noqa: BLE001
            print(f"  MISSING {ref:52} {type(exc).__name__}")
            missing.append(ref)

    if missing:
        print(f"\n{len(missing)} repo(s) would hit the network at run time:", file=sys.stderr)
        for repo in missing:
            print(f"  {repo}", file=sys.stderr)
        print("\nRun --download to fill them.", file=sys.stderr)
        return 1
    print(f"\nAll {len(repos)} repos resolve offline. Safe to mount read-only.")
    return 0


def check_locals(targets: list[str], root: Path) -> int:
    """Local checkpoints are shipped, not fetched, so the only question is
    whether they are actually there. A missing one fails at fit time exactly
    like a cache miss, so it belongs in the same gate."""
    missing = []
    for target in targets:
        ok, detail = check_local(target, root)
        print(f"  {'OK     ' if ok else 'MISSING'} {target:52} {detail if not ok else ''}")
        if not ok:
            missing.append(target)
    if missing:
        print(f"\n{len(missing)} local checkpoint(s) absent; those cards cannot load.",
              file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--catalog", type=Path, default=None,
                   help="model catalog JSON. Defaults to $AOB_MODEL_CATALOG, then "
                        "$SCENARIOS_DATA_DIR/shared/tsfm/model_catalog.json (the same file "
                        "CouchDB is seeded from), then the in-repo example.")
    p.add_argument("--revision", default=None,
                   help="pin every repo to this revision; omit to track each repo's default branch")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel file downloads per repo (default: 8)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="print per-repo size and exit (default)")
    mode.add_argument("--download", action="store_true", help="fill the cache; resumable")
    mode.add_argument("--check", action="store_true", help="offline: verify the cache is complete")
    p.add_argument("--status", default="active", metavar="STATUS",
                   help="only cards with this status, or 'any' for all. Default 'active', which "
                        "matches model_store.list_models, so the cache holds exactly what "
                        "find_models and search can discover. Deprecate a card to drop its "
                        "weights from the image without losing its lineage.")
    p.add_argument("--include", default=None, metavar="REGEX",
                   help="only act on repos matching this regex. Lets a Dockerfile fetch one "
                        "ecosystem per RUN, so the weights land in several layers that pull in "
                        "parallel instead of one 21 GB blob.")
    p.add_argument("--print-repos", action="store_true", help="print just the repo ids, one per line")
    args = p.parse_args()

    catalog, origin = resolve_catalog(args.catalog)
    if not catalog.is_file():
        print(f"no catalog at {catalog} (from {origin})", file=sys.stderr)
        if origin == "in-repo EXAMPLE":
            print("Set AOB_MODEL_CATALOG or SCENARIOS_DATA_DIR to your real catalog.",
                  file=sys.stderr)
        return 1
    if origin == "in-repo EXAMPLE":
        print(f"NOTE: using the in-repo EXAMPLE catalog at {catalog}.\n"
              f"      Set AOB_MODEL_CATALOG or SCENARIOS_DATA_DIR for the real one.\n")

    cards = load_cards(catalog)
    total_cards = len(cards)
    skipped_status: dict[str, int] = {}
    if args.status != "any":
        kept = []
        for c in cards:
            s = status_of(c)
            if s == args.status:
                kept.append(c)
            else:
                skipped_status[s] = skipped_status.get(s, 0) + 1
        cards = kept

    repos: list[str] = []
    locals_: list[str] = []
    notes: list[str] = []
    for card in cards:
        kind, target, why = classify(card)
        notes.append(why)
        if kind == "hub" and target and target not in repos:
            repos.append(target)
        elif kind == "local" and target and target not in locals_:
            locals_.append(target)

    if args.include:
        keep = re.compile(args.include)
        before = len(repos)
        repos = [r for r in repos if keep.search(r)]
        locals_ = [x for x in locals_ if keep.search(x)]
        notes.append(f"--include {args.include!r} kept {len(repos)} of {before} repos")

    if args.print_repos:
        print("\n".join(repos))
        return 0

    head = f"{total_cards} card(s) in {catalog} (from {origin})"
    if skipped_status:
        detail = ", ".join(f"{n} {s}" for s, n in sorted(skipped_status.items()))
        head += f"; {len(cards)} with status={args.status} ({detail} skipped)"
    print(f"{head}, {len(repos)} Hub repo(s), {len(locals_)} local checkpoint(s)\n")
    for n in notes:
        print(f"  {n}")
    print()

    # Only bail early when there is nothing of EITHER kind. A catalog of purely
    # local checkpoints still has to be verified; returning here skipped it.
    if not repos and not locals_:
        print("nothing to fetch and nothing to verify")
        return 0
    if args.revision and len(repos) > 1:
        print("note: --revision applies to every repo, which is rarely what you want with "
              "more than one; prefer pinning per card in the catalog.\n")

    root = Path(os.environ.get("AOB_HOME") or Path.cwd())

    if args.download:
        rc = download(repos, args.revision, args.workers) if repos else 0
        return max(rc, check_locals(locals_, root)) if locals_ else rc
    if args.check:
        print("cards:")
        rc = validate_cards(cards)
        if repos:
            rc = max(rc, check(repos, args.revision))
        return max(rc, check_locals(locals_, root))
    if locals_:
        print(f"{len(locals_)} local checkpoint(s), verified against {root}:")
        check_locals(locals_, root)
        print()
    return report(repos, args.revision) if repos else 0


if __name__ == "__main__":
    raise SystemExit(main())
