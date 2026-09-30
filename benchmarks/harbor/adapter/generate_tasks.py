"""Generate Harbor task directories from AssetOpsBench scenarios.

    python benchmarks/harbor/adapter/generate_tasks.py --overwrite

    python benchmarks/harbor/adapter/generate_tasks.py \
      --scenario-root <path-to>/scenarios_data \
      --profile benchmarks/scenario_suite/mini.yaml \
      --output-dir benchmarks/harbor/datasets/assetopsbench-mini \
      --overwrite

Without --scenario-root, tasks load the repo's own scenarios_data. Any other
root is an external suite, loaded from SUITE_DATA_DIR, where
overlays/private-data.yaml mounts its shared/ directory.

Task names come from category and scenario id only, so they stay stable across
runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_SCENARIO_ROOT = REPO_ROOT / "src/couchdb/scenarios_data"
# Where an external suite sits inside the task container: the per-task layer
# copies scenario_<id>/ here, and overlays/private-data.yaml mounts shared/.
SUITE_DATA_DIR = "/opt/suite/scenarios_data"
CATEGORIES = ("car", "fcc", "fmea", "fmsr", "health", "tsfm", "wosr")
TEMPLATE_FILES = (
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
)

# Every file evaluation.loader.load_scenario_dirs looks for in a scenario
# folder. question.txt and groundtruth.txt are required; the rest are optional
# scorer inputs, and scenario_meta.json is the one that selects the scorer.
SCENARIO_INPUT_FILES = (
    "question.txt",
    "groundtruth.txt",
    "groundtruth_eval.json",
    "scenario_meta.json",
    "rubric.json",
    "reference_answer.json",
)


def scoring_method_for(source: Path) -> str:
    """Mirror evaluation.loader: scenario_meta.json picks the scorer."""
    meta_path = source / "scenario_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and meta.get("scoring_method"):
            return str(meta["scoring_method"])
    return "static_json"


def data_load_inputs(source: Path) -> list[Path]:
    """Paths under a scenario folder that init_data.py reads, relative to it.

    manifest.json, plus any file or directory a manifest value names inside the
    folder. An allowlist, so an answer file under any name stays out of the
    agent's image. Paths into the sibling shared/ are left to the overlay.
    """
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scenario has no manifest.json: {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = source.resolve()
    inputs = [Path("manifest.json")]
    for value in manifest.values():
        # A value is a path, "default", inline documents, or a list of those.
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, str) or Path(item).is_absolute():
                continue
            candidate = (source / item).resolve()
            # Never the folder itself: that would copy the answers beside it.
            inside = candidate != root and candidate.is_relative_to(root)
            if inside and candidate.exists():
                inputs.append(candidate.relative_to(root))
    return list(dict.fromkeys(inputs))


def scenario_ids_by_category(profile_path: Path) -> list[tuple[str, str]]:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    pairs: list[tuple[str, str]] = []
    for category in CATEGORIES:
        for scenario_id in profile.get(category) or []:
            pairs.append((category, str(scenario_id)))
    return pairs


def generate(
    *,
    category: str,
    scenario_id: str,
    scenario_root: Path,
    template: Path,
    output_dir: Path,
    overwrite: bool,
    data_dir: str | None = None,
) -> Path:
    source = scenario_root / f"scenario_{scenario_id}"
    if not source.is_dir():
        raise FileNotFoundError(f"scenario folder not found: {source}")

    task_dir = output_dir / f"{category}-{scenario_id}"
    if task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{task_dir} exists; pass --overwrite")
        shutil.rmtree(task_dir)

    for relative in TEMPLATE_FILES:
        target = task_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (template / relative).read_text(encoding="utf-8")
        # The scenario id appears in the healthcheck command, the task name and
        # the COPY line, so a plain substitution covers the whole template.
        text = text.replace("scenario_1", f"scenario_{scenario_id}")
        text = text.replace("wosr-1", f"{category}-{scenario_id}")
        text = text.replace('scenario_id = "1"', f'scenario_id = "{scenario_id}"')
        text = text.replace(
            'AOB_SCENARIO_ID = "1"', f'AOB_SCENARIO_ID = "{scenario_id}"'
        )
        text = text.replace('category = "wosr"', f'category = "{category}"')
        text = text.replace(
            'scoring_method = "static_json"',
            f'scoring_method = "{scoring_method_for(source)}"',
        )
        text = text.replace("init_data.py 1", f"init_data.py {scenario_id}")
        # The template's description is scenario 1's; replace it wholesale.
        text = re.sub(
            r'^description = ".*"$',
            f'description = "AssetOpsBench scenario {scenario_id}, '
            f'{category} category."',
            text,
            flags=re.MULTILINE,
        )
        text = text.replace(
            '"assetopsbench", "wosr",', f'"assetopsbench", "{category}",'
        )
        if data_dir:
            # External suite: the scenario is copied to data_dir, beside the
            # mounted shared/, and only init_data.py reads it. The agent's
            # SCENARIOS_DATA_DIR stays on the repo copy.
            text = text.replace(
                "/opt/aob/src/couchdb/scenarios_data/", f"{data_dir.rstrip('/')}/"
            )
            text = text.replace(
                'command = "uv run python src/couchdb/init_data.py',
                f'command = "SCENARIOS_DATA_DIR={data_dir} '
                "uv run python src/couchdb/init_data.py",
            )
        target.write_text(text, encoding="utf-8")

    # The question the agent sees.
    shutil.copy(source / "question.txt", task_dir / "instruction.md")

    # Build context for the per-task image, the agent's container: only what
    # init_data.py reads. Answers go to tests/ and solution/ only.
    context = task_dir / "environment" / f"scenario_{scenario_id}"
    context.mkdir(parents=True)
    for relative in data_load_inputs(source):
        if (source / relative).is_dir():
            shutil.copytree(source / relative, context / relative, dirs_exist_ok=True)
        else:
            (context / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source / relative, context / relative)

    # Ground truth for the verifier: every file evaluation.loader reads, since
    # without scenario_meta.json a scenario silently falls back to static_json.
    verifier_scenarios = task_dir / "tests" / "scenarios" / f"scenario_{scenario_id}"
    verifier_scenarios.mkdir(parents=True, exist_ok=True)
    for name in SCENARIO_INPUT_FILES:
        candidate = source / name
        if candidate.exists():
            shutil.copy(candidate, verifier_scenarios / name)
    for name in ("test.sh", "to_reward.py"):
        shutil.copy(template / "tests" / name, task_dir / "tests" / name)
    (task_dir / "tests" / "test.sh").chmod(0o755)

    # Oracle.
    solution = task_dir / "solution"
    solution.mkdir(parents=True, exist_ok=True)
    for name in ("question.txt", "groundtruth.txt"):
        shutil.copy(source / name, solution / name)
    solve = (template / "solution" / "solve.sh").read_text(encoding="utf-8")
    solve = solve.replace('"oracle_1"', f'"oracle_{scenario_id}"')
    solve = solve.replace('"scenario_id": "1"', f'"scenario_id": "{scenario_id}"')
    solve = solve.replace(
        "/logs/agent/oracle_1.json", f"/logs/agent/oracle_{scenario_id}.json"
    )
    (solution / "solve.sh").write_text(solve, encoding="utf-8")
    (solution / "solve.sh").chmod(0o755)

    return task_dir


def write_dataset_files(
    *, output_dir: Path, dataset_name: str, task_dirs: list[Path]
) -> None:
    """Emit dataset.toml and metric.py beside the generated tasks.

    Task digests are left to `harbor dataset add`.
    """
    shutil.copy(REPO_ROOT / "benchmarks/harbor/metric.py", output_dir / "metric.py")

    names = "\n".join(f"#   harbor dataset add {d.name}" for d in task_dirs)
    (output_dir / "dataset.toml").write_text(
        "# Generated by benchmarks/harbor/adapter/generate_tasks.py. Task digests\n"
        "# are added by `harbor dataset add <task-dir>`.\n"
        'schema_version = "1.0"\n\n'
        "[dataset]\n"
        f'name = "{dataset_name}"\n'
        'version = "0.1.0"\n'
        'description = "AssetOpsBench scenarios as Harbor tasks."\n'
        'authors = [{ name = "AssetOpsBench Team" }]\n'
        'keywords = ["industrial", "asset-operations", "mcp", "tool-use", "agents"]\n\n'
        "# Add each task, then publish:\n"
        f"{names}\n\n"
        "[[files]]\n"
        'path = "metric.py"\n',
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=None,
        help="Scenario folders to generate from. Default: the repo's "
        "src/couchdb/scenarios_data. Any other root is an external suite, "
        f"loaded from {SUITE_DATA_DIR}: run with --extra-docker-compose "
        "benchmarks/harbor/overlays/private-data.yaml.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "benchmarks/scenario_suite/open.yaml",
    )
    parser.add_argument(
        "--template", type=Path, default=REPO_ROOT / "benchmarks/harbor/template"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks/harbor/datasets/assetopsbench-open",
    )
    parser.add_argument(
        "--dataset-name",
        default="assetopsbench/open",
        help="Harbor dataset name written into dataset.toml.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and skip profile scenarios with no folder under "
        "--scenario-root, instead of failing.",
    )
    args = parser.parse_args()

    # Naming the repo's own folder explicitly still means the repo copy.
    external = (
        args.scenario_root is not None
        and args.scenario_root.resolve() != REPO_SCENARIO_ROOT.resolve()
    )
    args.scenario_root = args.scenario_root or REPO_SCENARIO_ROOT
    data_dir = SUITE_DATA_DIR if external else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    skipped = []
    for category, scenario_id in scenario_ids_by_category(args.profile):
        if (
            args.skip_missing
            and not (args.scenario_root / f"scenario_{scenario_id}").is_dir()
        ):
            skipped.append(f"{category}-{scenario_id}")
            continue
        written.append(
            generate(
                category=category,
                scenario_id=scenario_id,
                scenario_root=args.scenario_root,
                template=args.template,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                data_dir=data_dir,
            )
        )
    if skipped:
        print(
            f"skipped {len(skipped)} scenario(s) missing from "
            f"{args.scenario_root}: {', '.join(skipped)}",
            file=sys.stderr,
        )

    write_dataset_files(
        output_dir=args.output_dir, dataset_name=args.dataset_name, task_dirs=written
    )

    for path in written:
        print(path)
    print(f"{len(written)} tasks written to {args.output_dir}")
    print("next: harbor run -p", args.output_dir, "--agent oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
