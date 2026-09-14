#!/usr/bin/env python3
"""Preflight for a skill library inside AssetOpsBench.

Run this after installing, before spending a benchmark run. It answers the one
question that matters at handoff: will the agent actually see the skills.

    python skills/preflight.py --assetops . --skills skills/repositories

Seven checks, in the order that a failure would block you:

  1. Skill tree     the collection is present, well-formed, and countable
  2. Patch          the Stirrup plug is applied to the target checkout
  3. Import         `skills_mount` imports and exposes the expected contract
  4. Mount k0       mounts nothing and appends nothing, so the baseline is intact
  5. Mount k1       copies the tree into a workspace and returns a prompt block
  6. Router         the mounted tree's entry point and router resolve
  7. Runner wiring  StirrupAgentRunner accepts `skills_dir` and `k_level`

Exit codes: 0 ready to run, 1 a check failed, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROWS: list[tuple[str, str, str]] = []


def ok(name: str, detail: str = "") -> None:
    ROWS.append(("PASS", name, detail))


def bad(name: str, detail: str) -> None:
    ROWS.append(("FAIL", name, detail))


def warn(name: str, detail: str) -> None:
    ROWS.append(("WARN", name, detail))


def check_tree(skills: pathlib.Path) -> int:
    graphs_dir = skills / "repo-skills"
    router = skills / "repo-skills-router" / "SKILL.md"
    if not graphs_dir.is_dir():
        bad("1 skill tree", f"no repo-skills directory under {skills}")
        return 0
    graphs = [p for p in graphs_dir.iterdir() if p.is_dir() and (p / "SKILL.md").exists()]
    subs = sum(len(list((g / "sub-skills").glob("*/SKILL.md"))) for g in graphs)
    total = len(graphs) + subs
    if not router.exists():
        bad("1 skill tree", "repo-skills-router/SKILL.md missing, routing will not work")
        return total
    entry = skills / "repo-skills-router" / "references" / "entry.md"
    if not entry.exists():
        warn("1 skill tree", "router references/entry.md missing; agents will route without the one-page entry")
    ok("1 skill tree", f"{len(graphs)} graphs, {total} skills, router present")
    # The mount copies every SKILL.md, which is the graph and sub-skill count
    # plus the router itself. Return the file count so check 5 compares like
    # with like.
    return total + 1


def check_patch(aob: pathlib.Path) -> bool:
    mount = aob / "src" / "agent" / "stirrup_agent" / "skills_mount.py"
    runner = aob / "src" / "agent" / "stirrup_agent" / "runner.py"
    if not runner.exists():
        bad("2 patch", f"not an AssetOpsBench checkout: {runner} missing")
        return False
    if not mount.exists():
        bad("2 patch", "skills_mount.py missing; apply patches/stirrup_skills_plug.diff")
        return False
    body = runner.read_text(errors="replace")
    missing = [t for t in ("skills_mount", "skills_dir", "k_level") if t not in body]
    if missing:
        bad("2 patch", f"runner.py lacks {missing}; the patch is not applied")
        return False
    ok("2 patch", "skills_mount.py present and runner.py wired")
    return True


def check_import(aob: pathlib.Path):
    sys.path.insert(0, str(aob / "src" / "agent" / "stirrup_agent"))
    try:
        import skills_mount  # type: ignore
    except Exception as exc:  # noqa: BLE001
        bad("3 import", f"{type(exc).__name__}: {exc}")
        return None
    for attr in ("mount_skills", "K_LEVELS"):
        if not hasattr(skills_mount, attr):
            bad("3 import", f"skills_mount has no `{attr}`")
            return None
    ok("3 import", f"K_LEVELS = {tuple(skills_mount.K_LEVELS)}")
    return skills_mount


def check_mounts(sm, skills: pathlib.Path, total: int) -> None:
    with tempfile.TemporaryDirectory() as td:
        ws0 = pathlib.Path(td) / "k0"
        ws0.mkdir()
        try:
            block = sm.mount_skills(skills, ws0, k_level="k0", code_backend="docker")
        except Exception as exc:  # noqa: BLE001
            bad("4 mount k0", f"{type(exc).__name__}: {exc}")
            return
        if block is not None:
            bad("4 mount k0", "k0 returned a prompt block; the baseline is not clean")
        elif any(ws0.iterdir()):
            bad("4 mount k0", "k0 wrote files into the workspace")
        else:
            ok("4 mount k0", "nothing mounted, nothing appended")

        ws1 = pathlib.Path(td) / "k1"
        ws1.mkdir()
        try:
            block = sm.mount_skills(skills, ws1, k_level="k1", code_backend="docker")
        except Exception as exc:  # noqa: BLE001
            bad("5 mount k1", f"{type(exc).__name__}: {exc}")
            return
        landed = list((ws1 / "skills").rglob("SKILL.md"))
        if not block:
            bad("5 mount k1", "no prompt block returned")
        elif not landed:
            bad("5 mount k1", "no SKILL.md landed in the workspace")
        else:
            if len(landed) != total:
                warn("5 mount k1", f"{len(landed)} SKILL.md landed, tree has {total}")
            ok("5 mount k1", f"{len(landed) - 1} skills plus the router mounted, "
                             f"prompt block {len(block)} chars")

        router = ws1 / "skills" / "repo-skills-router" / "SKILL.md"
        if not router.exists():
            bad("6 router", "router did not survive the mount")
        elif "/workspace/skills" not in (block or ""):
            bad("6 router", "prompt block does not name the docker mount path")
        else:
            ok("6 router", "router mounted and named in the prompt block")


def check_runner(aob: pathlib.Path) -> None:
    """Parse runner.py rather than importing it, so no heavy deps are needed."""
    src = (aob / "src" / "agent" / "stirrup_agent" / "runner.py").read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        bad("7 runner wiring", f"runner.py does not parse: {exc}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StirrupAgentRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = {a.arg for a in item.args.args} | {a.arg for a in item.args.kwonlyargs}
                    missing = {"skills_dir", "k_level"} - args
                    if missing:
                        bad("7 runner wiring", f"__init__ lacks {sorted(missing)}")
                    else:
                        ok("7 runner wiring", "StirrupAgentRunner accepts skills_dir and k_level")
                    return
    bad("7 runner wiring", "StirrupAgentRunner.__init__ not found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assetops", type=pathlib.Path, required=True,
                    help="path to the AssetOpsBench checkout with the patch applied")
    ap.add_argument("--skills", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "repositories",
                    help="path to the skill collection (the directory holding repo-skills/)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    total = check_tree(a.skills.resolve())
    if check_patch(a.assetops.resolve()):
        sm = check_import(a.assetops.resolve())
        if sm is not None:
            check_mounts(sm, a.skills.resolve(), total)
        check_runner(a.assetops.resolve())

    failed = any(r[0] == "FAIL" for r in ROWS)
    if a.json:
        print(json.dumps({"ready": not failed,
                          "checks": [{"status": s, "check": c, "detail": d} for s, c, d in ROWS]},
                         indent=2))
    else:
        for status, name, detail in ROWS:
            print(f"{status:<5} {name:<18} {detail}")
        print()
        print("READY: run the benchmark" if not failed
              else "NOT READY: fix the failures above before spending a run")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
