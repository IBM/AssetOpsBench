#!/usr/bin/env python3
"""Static and leakage gates for an AssetOpsBench skill library.

Run this against any library before mounting it, whether it is the reference
library in this repository or one you built yourself. It checks the contract in
`skills/CONTRACT.md`: frontmatter, per-graph licence consistency,
self-containment, and the industrial axes.

Gate 3 is the one specific to a benchmark. A skill library sits closer to the
answers than anything else an agent reads, so a `leakage-class: solution` skill
fails outright and any eight-word sequence shared with the answer set is a
failure that names the scenario it came from.

    python skills/tools/validate_skills.py --root skills/repositories

The answer set is not in the repository: `benchmarks/scenario_suite/*.yaml` hold
scenario ids only, so an audit pointed at the checkout proves nothing. Point it
at where the answers actually live, by any of three routes:

    # a file the evaluation harness exported
    ... --answers /path/to/scenarios_with_answers.jsonl

    # a directory of them, walked recursively
    ... --answers-dir /path/to/exported_answers/

    # the published dataset, every config and split by default
    ... --answers-hf ibm-research/AssetOpsBench

Exit codes: 0 all gates pass, 1 a gate failed, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REQUIRED_FIELDS = ("name", "description", "license", "metadata")
# Industrial extension to the AREX contract. Domain skills carry the index axes
# and the leakage class; tool-surface skills do not need the asset axis.
CAPABILITY_FAMILIES = {f"C{i}" for i in range(1, 13)}
ASSET_CLASSES = {
    "A0", "chiller-hvac", "ahu", "pumps", "motors-drives", "fans-blowers",
    "compressors", "bearings-gearboxes", "wind-turbine", "transformers-electrical",
}
LEAKAGE_CLASSES = {"ops", "solution"}
LEAK_PATTERNS = [
    (re.compile(r"/home/[a-z0-9_.-]+/", re.I), "absolute home path"),
    (re.compile(r"/Users/[a-z0-9_.-]+/", re.I), "absolute macOS home path"),
    (re.compile(r"site-packages"), "installed-package path"),
    (re.compile(r"conda activate|micromamba activate|source .*/bin/activate"), "environment activation"),
    (re.compile(r"\.disco/agent"), "DisCo managed path"),
]
FORBIDDEN_EVIDENCE = [
    "benchmarks/scenario_suite",
    "src/evaluation/scorers",
    "src/scenarios/",
]
#: Populated in main() from whichever answer source was given; None means no
#: source was supplied, which is a warning rather than a pass.
_ANSWER_BLOBS: list[tuple[str, str]] | None = None
DEBRIS = ("__pycache__", ".pyc", ".ipynb_checkpoints", ".DS_Store")
ROOT_LINES = (80, 150)
SUB_LINES = (80, 250)


def parse_frontmatter(text: str) -> tuple[dict, str] | tuple[None, str]:
    if not text.startswith("---\n"):
        return None, "no frontmatter block"
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, "unterminated frontmatter block"
    block = text[4:end]
    data: dict = {}
    key = None
    # A double-quoted YAML scalar may span lines. Join continuations onto the
    # value before parsing, otherwise a perfectly valid multi-line description
    # is reported as unquoted, which is a bug in the checker and not in the
    # skill it is checking.
    lines, joined = block.split("\n"), []
    for line in lines:
        if (joined and isinstance(joined[-1], str) and line.startswith("  ")
                and joined[-1].count('"') == 1 and '"' in joined[-1]):
            joined[-1] = joined[-1].rstrip() + " " + line.strip()
            continue
        joined.append(line)
    for line in joined:
        if not line.strip():
            continue
        if line.startswith("  ") and key:
            k, _, v = line.strip().partition(":")
            data.setdefault(key, {})
            if isinstance(data[key], dict):
                data[key][k.strip()] = v.strip()
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        data[key] = v.strip() if v.strip() else {}
    return data, ""


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, where: str, msg: str) -> None:
        self.rows.append((level, where, msg))

    @property
    def failed(self) -> bool:
        return any(r[0] == "FAIL" for r in self.rows)

    def print(self) -> None:
        for level, where, msg in self.rows:
            print(f"{level:<5} {where}: {msg}")
        fails = sum(1 for r in self.rows if r[0] == "FAIL")
        warns = sum(1 for r in self.rows if r[0] == "WARN")
        print(f"\n{fails} failures, {warns} warnings, {len(self.rows)} findings")


def _tree_of(skill_md: pathlib.Path, root: pathlib.Path) -> str:
    """The skill graph a file belongs to: <collection>/<graph-id>."""
    rel = skill_md.relative_to(root).parts
    return "/".join(rel[:2]) if len(rel) > 1 else rel[0]


def gate_frontmatter(root: pathlib.Path, rep: Report) -> None:
    """Gate 1: frontmatter contract and per-tree licence consistency."""
    licences: dict[str, set[str]] = {}
    for skill_md in sorted(root.rglob("SKILL.md")):
        rel = skill_md.relative_to(root).as_posix()
        text = skill_md.read_text(encoding="utf-8")
        fm, err = parse_frontmatter(text)
        if fm is None:
            rep.add("FAIL", rel, err)
            continue
        for field in REQUIRED_FIELDS:
            if field not in fm:
                rep.add("FAIL", rel, f"missing required frontmatter field `{field}`")
        name = fm.get("name", "")
        if name != skill_md.parent.name:
            rep.add("FAIL", rel, f"name `{name}` does not equal directory `{skill_md.parent.name}`")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", str(name)):
            rep.add("FAIL", rel, f"name `{name}` violates the id pattern")
        desc = fm.get("description", "")
        if not (isinstance(desc, str) and desc.startswith('"') and desc.rstrip().endswith('"')):
            rep.add("FAIL", rel, "description must be a double-quoted string")
        lic = fm.get("license", "")
        if not isinstance(lic, str) or not lic.strip():
            rep.add("FAIL", rel, "license must be a non-empty single-line value")
        else:
            licences.setdefault(_tree_of(skill_md, root), set()).add(lic.strip())
        meta = fm.get("metadata", {})
        role = meta.get("disco-role") if isinstance(meta, dict) else None
        if role != "operating":
            rep.add("FAIL", rel, f"metadata.disco-role must be `operating`, found `{role}`")
        # Industrial extension: any skill declaring a capability family must
        # declare a valid asset class and a leakage class, and only `ops` ships.
        if isinstance(meta, dict) and "capability-family" in meta:
            fams = {x.strip() for x in str(meta["capability-family"]).split(",")}
            bad = fams - CAPABILITY_FAMILIES
            if bad:
                rep.add("FAIL", rel, f"unknown capability family: {sorted(bad)}")
            classes = {x.strip() for x in str(meta.get("asset-class", "")).split(",")}
            bad = classes - ASSET_CLASSES
            if bad:
                rep.add("FAIL", rel, f"unknown asset class: {sorted(bad)}")
            lk = str(meta.get("leakage-class", "")).strip()
            if lk not in LEAKAGE_CLASSES:
                rep.add("FAIL", rel, f"leakage-class must be one of {sorted(LEAKAGE_CLASSES)}, found `{lk}`")
            elif lk == "solution":
                rep.add("FAIL", rel, "a `solution` class skill must never ship to an evaluated agent")

        is_router = skill_md.parent.name == "repo-skills-router"
        dmi = str(fm.get("disable-model-invocation", "")).lower()
        if is_router and dmi == "true":
            rep.add("FAIL", rel, "the router must not set disable-model-invocation")
        if not is_router and dmi != "true":
            rep.add("FAIL", rel, "disable-model-invocation: true is required")

        n = len(text.splitlines())
        lo, hi = ROOT_LINES if skill_md.parent.parent.name == "repo-skills" else SUB_LINES
        if n > hi:
            rep.add("WARN", rel, f"{n} lines exceeds the {hi}-line target; move detail to references/")
        elif n < lo and not is_router:
            rep.add("WARN", rel, f"{n} lines is below the {lo}-line target; likely underspecified")

    for tree, lics in sorted(licences.items()):
        if len(lics) > 1:
            rep.add("FAIL", tree,
                    f"inconsistent licences within one skill tree: {sorted(lics)}")


def gate_static(root: pathlib.Path, rep: Report) -> None:
    """Gate 2: self-containment, leakage of local paths, artifact debris."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if any(d in rel for d in DEBRIS):
            rep.add("FAIL", rel, "build or editor debris inside the runtime tree")
            continue
        if not path.is_file() or path.suffix not in {".md", ".py", ".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in LEAK_PATTERNS:
            m = pattern.search(text)
            if m:
                rep.add("FAIL", rel, f"{label} leaked into a runtime file: {m.group(0)!r}")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if link.startswith(("http://", "https://", "#")):
                continue
            target = (path.parent / link).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                rep.add("FAIL", rel, f"link escapes the skill tree: {link}")
                continue
            if not target.exists():
                rep.add("FAIL", rel, f"broken link: {link}")


def _record_label(obj, index: int) -> str:
    """A name for an answer record, so a leakage hit can be triaged rather than
    only counted."""
    if isinstance(obj, dict):
        for key in ("id", "scenario_id", "task_id", "name", "uid", "utterance_id"):
            if key in obj:
                return f"{key}={obj[key]}"
    return f"record#{index}"


def load_answer_blobs(answers: pathlib.Path | None,
                      answers_dir: pathlib.Path | None,
                      hf_dataset: str | None,
                      hf_configs: list[str] | None,
                      hf_split: str | None,
                      rep: Report) -> list[tuple[str, str]] | None:
    """Collect the benchmark's answer text from wherever it actually lives.

    Three sources, because the answers are not in the repository. The in-repo
    `benchmarks/scenario_suite/*.yaml` files hold scenario ids only, so an audit
    pointed at the checkout proves nothing. The real surfaces are the published
    dataset and whatever export the evaluation harness writes.

    Returns a list of (label, text), or None if no source was given.
    """
    blobs: list[tuple[str, str]] = []

    def eat_file(p: pathlib.Path) -> None:
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix == ".jsonl":
            for i, line in enumerate(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    blobs.append((f"{p.name}:{_record_label(obj, i)}", json.dumps(obj)))
                except json.JSONDecodeError:
                    blobs.append((f"{p.name}:line{i}", line))
        elif p.suffix == ".json":
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                blobs.append((p.name, raw))
                return
            if isinstance(obj, list):
                for i, o in enumerate(obj):
                    blobs.append((f"{p.name}:{_record_label(o, i)}", json.dumps(o)))
            else:
                blobs.append((p.name, json.dumps(obj)))
        else:
            blobs.append((p.name, raw))

    if answers is not None:
        if not answers.exists():
            rep.add("FAIL", "collection", f"answers file not found: {answers}")
            return []
        eat_file(answers)

    if answers_dir is not None:
        if not answers_dir.is_dir():
            rep.add("FAIL", "collection", f"answers directory not found: {answers_dir}")
            return []
        found = [p for p in sorted(answers_dir.rglob("*"))
                 if p.is_file() and p.suffix in {".json", ".jsonl", ".yaml", ".yml",
                                                 ".txt", ".csv", ".md"}]
        if not found:
            rep.add("FAIL", "collection", f"no answer files under {answers_dir}")
            return []
        for p in found:
            eat_file(p)

    if hf_dataset is not None:
        try:
            from datasets import get_dataset_config_names, load_dataset
        except ImportError:
            rep.add("FAIL", "collection",
                    "--answers-hf needs the `datasets` package; "
                    "install it with: pip install datasets")
            return []
        try:
            configs = hf_configs or list(get_dataset_config_names(hf_dataset))
        except Exception as exc:  # noqa: BLE001
            rep.add("FAIL", "collection",
                    f"could not list configs of {hf_dataset}: "
                    f"{type(exc).__name__}: {exc}")
            return []
        if not configs:
            configs = [None]
        for cfg in configs:
            try:
                ds = load_dataset(hf_dataset, cfg) if cfg else load_dataset(hf_dataset)
            except Exception as exc:  # noqa: BLE001
                rep.add("FAIL", "collection",
                        f"could not load {hf_dataset} config {cfg}: "
                        f"{type(exc).__name__}: {exc}")
                continue
            splits = [hf_split] if hf_split else list(ds.keys())
            for sp in splits:
                if sp not in ds:
                    continue
                for i, row in enumerate(ds[sp]):
                    blobs.append((f"{hf_dataset}/{cfg}/{sp}:{_record_label(row, i)}",
                                  json.dumps(row, default=str)))

    if answers is None and answers_dir is None and hf_dataset is None:
        return None
    return blobs


def gate_leakage(root: pathlib.Path, answers: pathlib.Path | None, rep: Report) -> None:
    """Gate 3: no benchmark answer content, and no evidence from excluded paths."""
    runtime_text: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".py", ".json", ".jsonl"}:
            runtime_text[path.relative_to(root).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace")

    # 3a: excluded evidence paths must not be cited by any runtime instruction.
    for rel, text in runtime_text.items():
        if rel.endswith("repo-provenance.md"):
            continue  # provenance records the exclusion itself
        for bad in FORBIDDEN_EVIDENCE:
            if bad in text:
                rep.add("FAIL", rel, f"cites an excluded evidence path: {bad}")

    # 3b: n-gram overlap with the answer set, when one is supplied.
    blobs = _ANSWER_BLOBS
    if blobs is None:
        rep.add("WARN", "collection",
                "no answer source supplied (--answers, --answers-dir or "
                "--answers-hf); the n-gram leakage audit did not run")
        return
    if not blobs:
        return  # the loader already recorded why

    def shingles(s: str, k: int = 8) -> set[str]:
        words = re.findall(r"[a-z0-9_]+", s.lower())
        return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}

    # Keep the owning record for each shingle, so a hit names the scenario it
    # came from. A leakage failure that cannot be traced back gets argued with
    # instead of fixed.
    owner: dict[str, str] = {}
    for label, b in blobs:
        for sh in shingles(b):
            owner.setdefault(sh, label)

    rep.add("INFO", "collection",
            f"leakage audit ran against {len(blobs)} answer records, "
            f"{len(owner)} distinct eight-word sequences")

    for rel, text in runtime_text.items():
        hits = shingles(text) & owner.keys()
        if hits:
            sample = sorted(hits)[:3]
            sources = sorted({owner[h] for h in hits})[:3]
            rep.add("FAIL", rel,
                    f"{len(hits)} eight-word sequences shared with the answer set "
                    f"(from {sources}), e.g. {sample}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("skills/repositories"))
    ap.add_argument("--answers", type=pathlib.Path, default=None,
                    help="scenario file containing reference answers, for the leakage audit")
    ap.add_argument("--answers-dir", type=pathlib.Path, default=None,
                    help="directory of answer files to audit against, walked recursively")
    ap.add_argument("--answers-hf", default=None, metavar="REPO_ID",
                    help="HuggingFace dataset holding the answers, "
                         "e.g. ibm-research/AssetOpsBench; needs `pip install datasets`")
    ap.add_argument("--hf-config", action="append", default=None, metavar="NAME",
                    help="restrict --answers-hf to this config; repeatable, "
                         "default is every config the dataset publishes")
    ap.add_argument("--hf-split", default=None,
                    help="restrict --answers-hf to this split, default is every split")
    a = ap.parse_args()
    if not a.root.is_dir():
        print(f"root not found: {a.root}", file=sys.stderr)
        return 2

    rep = Report()
    global _ANSWER_BLOBS
    _ANSWER_BLOBS = load_answer_blobs(a.answers, a.answers_dir, a.answers_hf,
                                      a.hf_config, a.hf_split, rep)
    gate_frontmatter(a.root, rep)
    gate_static(a.root, rep)
    gate_leakage(a.root, a.answers, rep)
    rep.print()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
