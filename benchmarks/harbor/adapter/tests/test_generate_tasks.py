"""The generated task must not hand the agent the answer.

An agent that can read groundtruth.txt scores 1.0 without touching CouchDB,
exactly as solution/solve.sh does. That failure is invisible in results: it
produces a perfect score, not an error, so only a check like this catches it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / "generate_tasks.py"
REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILE = REPO_ROOT / "benchmarks/scenario_suite/open.yaml"

ANSWER_FILES = (
    "groundtruth.txt",
    "groundtruth_eval.json",
    "reference_answer.json",
    "rubric.json",
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("generate_tasks", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_tasks"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    pytest.importorskip("yaml")
    if not PROFILE.is_file():
        pytest.skip(f"no profile at {PROFILE}")
    module = _load_adapter()
    out = tmp_path_factory.mktemp("tasks")
    argv = sys.argv
    sys.argv = [
        "generate_tasks.py",
        "--profile", str(PROFILE),
        "--output-dir", str(out),
        "--overwrite",
    ]
    try:
        assert module.main() == 0
    finally:
        sys.argv = argv
    dirs = sorted(p for p in out.iterdir() if p.is_dir())
    assert dirs, "the adapter wrote no tasks"
    return dirs


def test_agent_image_has_no_answer_files(tasks):
    """environment/ becomes the agent's image. No answer may reach it."""
    leaked = [
        str(path.relative_to(task.parent))
        for task in tasks
        for path in (task / "environment").rglob("*")
        if path.name in ANSWER_FILES
    ]
    assert not leaked, (
        "answer files in the agent's build context, where an agent can read "
        "them and score without doing the task:\n  " + "\n  ".join(leaked)
    )


def test_agent_image_keeps_the_manifest(tasks):
    """init_data.py still needs manifest.json from the scenario folder."""
    for task in tasks:
        scenario_dirs = list((task / "environment").glob("scenario_*"))
        assert scenario_dirs, f"{task.name}: no scenario folder in the build context"
        for scenario in scenario_dirs:
            assert (scenario / "manifest.json").is_file(), (
                f"{task.name}/{scenario.name}: manifest.json was withheld; "
                "the data load reads it"
            )


def test_verifier_and_oracle_still_get_the_answer(tasks):
    """The two copies that are supposed to exist."""
    for task in tasks:
        assert (task / "solution" / "groundtruth.txt").is_file(), (
            f"{task.name}: the oracle needs solution/groundtruth.txt"
        )
        scored = list((task / "tests" / "scenarios").glob("scenario_*/groundtruth.txt"))
        assert scored, (
            f"{task.name}: the verifier needs tests/scenarios/*/groundtruth.txt"
        )
