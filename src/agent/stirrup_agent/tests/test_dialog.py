"""Multi-turn dialog: loading, the turn mount, the router, and the M arms.

The failure these guard against: a turn that silently runs without its history
while the run is still labelled ``m1``. That is the same class of error as a k1
run that mounted nothing, and it is just as invisible in the results table.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent.stirrup_agent.dialog import (
    Dialog,
    DialogTurn,
    load_dialog,
    run_dialog,
)
from agent.stirrup_agent.turns_mount import (
    TurnRecord,
    build_router,
    resolve_m_level,
    stage_turns,
    turns_prompt,
)


# -- the on-disk dialog format ---------------------------------------------


def _scenario(tmp_path: Path, turns: list[str]) -> Path:
    root = tmp_path / "scenario_7"
    root.mkdir()
    (root / "question.txt").write_text(turns[0])
    (root / "groundtruth.txt").write_text("42")
    for i, text in enumerate(turns[1:], start=2):
        turn_dir = root / "turns" / f"{i:02d}"
        turn_dir.mkdir(parents=True)
        (turn_dir / "question.txt").write_text(text)
    return root


def test_existing_single_turn_scenario_loads_as_a_one_turn_dialog(
    tmp_path: Path,
) -> None:
    """No turns/ directory means the whole current suite already loads."""
    root = _scenario(tmp_path, ["How many work orders?"])

    dialog = load_dialog(root)

    assert dialog.is_single_turn
    assert dialog.id == "7"
    assert dialog.turns[0].text == "How many work orders?"
    assert dialog.turns[0].expected_answer == "42"


def test_turns_directory_extends_the_dialog(tmp_path: Path) -> None:
    root = _scenario(
        tmp_path, ["How many work orders?", "How many are open?", "As a percentage?"]
    )

    dialog = load_dialog(root)

    assert [t.n for t in dialog.turns] == [1, 2, 3]
    assert dialog.turns[2].text == "As a percentage?"
    assert not dialog.is_single_turn


def test_depends_on_is_read(tmp_path: Path) -> None:
    root = _scenario(tmp_path, ["one", "two"])
    (root / "turns" / "02" / "depends_on.txt").write_text("1")

    assert load_dialog(root).turns[1].depends_on == [1]


def test_a_gap_in_turn_numbers_raises(tmp_path: Path) -> None:
    root = _scenario(tmp_path, ["one"])
    gap = root / "turns" / "03"
    gap.mkdir(parents=True)
    (gap / "question.txt").write_text("three")

    with pytest.raises(ValueError, match="contiguous"):
        load_dialog(root)


def test_turn_01_directory_is_rejected(tmp_path: Path) -> None:
    """Turn 1 is question.txt. Two sources for it would diverge."""
    root = _scenario(tmp_path, ["one"])
    dup = root / "turns" / "01"
    dup.mkdir(parents=True)
    (dup / "question.txt").write_text("also one")

    with pytest.raises(ValueError, match="start at 02"):
        load_dialog(root)


# -- the M control ----------------------------------------------------------


def test_m0_never_mounts() -> None:
    assert resolve_m_level("m0", 1) is False
    assert resolve_m_level("m0", 5) is False


def test_m1_mounts_from_turn_two() -> None:
    assert resolve_m_level("m1", 1) is False
    assert resolve_m_level("m1", 2) is True


def test_a_bad_m_level_raises() -> None:
    with pytest.raises(ValueError, match="m_level must be one of"):
        resolve_m_level("m2", 2)


def test_turn_one_gets_no_prompt_block() -> None:
    assert turns_prompt(1, m_level="m1") is None


def test_prompt_names_the_router_at_the_mount() -> None:
    block = turns_prompt(2, m_level="m1", code_backend="docker")
    assert "/workspace/turns/turn-router/SKILL.md" in block
    assert turns_prompt(2, m_level="m1", code_backend="local").startswith(
        "This is turn 2"
    )


def test_m1_full_drops_the_routing_discipline() -> None:
    routed = turns_prompt(2, m_level="m1")
    full = turns_prompt(2, m_level="m1-full")
    assert "Route before you act" in routed
    assert "Route before you act" not in full


# -- staging and the router -------------------------------------------------


@pytest.fixture
def records(tmp_path: Path) -> list[TurnRecord]:
    ws1 = tmp_path / "turn-01"
    ws1.mkdir()
    (ws1 / "counts.csv").write_text("site,count\nMAIN,39\n")
    return [
        TurnRecord(
            n=1,
            ask="How many work orders are logged at the main site?",
            answer="39",
            workspace=ws1,
            tool_calls=["wo__count_work_orders"],
            duration_ms=1234.0,
        )
    ]


def test_staging_lays_out_what_the_router_promises(
    records: list[TurnRecord], tmp_path: Path
) -> None:
    staged = stage_turns(records, tmp_path / "_staged" / "turn-02", current_turn=2)

    assert (staged / "turn-router" / "SKILL.md").is_file()
    assert (staged / "turn-01" / "ASK.md").is_file()
    assert (staged / "turn-01" / "ANSWER.md").read_text().strip() == "39"
    assert (staged / "turn-01" / "workspace" / "counts.csv").is_file()
    assert json.loads((staged / "turn-01" / "turn.json").read_text())["n"] == 1


def test_staging_does_not_nest_earlier_mounts(tmp_path: Path) -> None:
    """Turn N-1's own mounts must not ride along into turn N.

    Without this the tree grows quadratically: turn 4 would carry turn 3's copy
    of turn 2's copy of turn 1.
    """
    ws = tmp_path / "turn-02"
    (ws / "turns" / "turn-01").mkdir(parents=True)
    (ws / "turns" / "turn-01" / "ANSWER.md").write_text("stale")
    (ws / "skills" / "repo-skills").mkdir(parents=True)
    (ws / "real_output.csv").write_text("kept")

    staged = stage_turns(
        [TurnRecord(n=2, ask="q", answer="a", workspace=ws)],
        tmp_path / "_staged",
        current_turn=3,
    )

    assert (staged / "turn-02" / "workspace" / "real_output.csv").is_file()
    assert not (staged / "turn-02" / "workspace" / "turns").exists()
    assert not (staged / "turn-02" / "workspace" / "skills").exists()


def test_router_lists_every_turn_with_its_ask(records: list[TurnRecord]) -> None:
    router = build_router(records, current_turn=2)

    assert "| 1 |" in router
    assert "How many work orders are logged at the main site?" in router
    assert "`turn-01/ANSWER.md`" in router
    assert "`turn-01/workspace/`" in router
    assert "wo__count_work_orders" in router


def test_router_marks_a_failed_turn(tmp_path: Path) -> None:
    router = build_router(
        [TurnRecord(n=1, ask="q", answer="Turn failed: X", failed=True)],
        current_turn=2,
    )
    assert "(failed)" in router


def test_router_escapes_a_pipe_in_the_ask() -> None:
    """A pipe in the question must not split the router's table row."""
    import re

    router = build_router(
        [TurnRecord(n=1, ask="count a | b", answer="ok")], current_turn=2
    )
    table_row = [line for line in router.splitlines() if line.startswith("| 1 |")][0]

    assert r"count a \| b" in table_row
    # Four columns means five unescaped delimiters, escaped ones excluded.
    assert len(re.findall(r"(?<!\\)\|", table_row)) == 5


# -- the outer loop ---------------------------------------------------------


@dataclass
class _FakeRunner:
    """Records what each turn was handed, and writes a file like a real turn."""

    turn: int
    workspace: Path
    turns_dir: Path | None
    seen: list[dict] = field(default_factory=list)

    async def run(self, question: str):
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / f"out-{self.turn}.txt").write_text(f"turn {self.turn}")
        self.seen.append({"turn": self.turn, "question": question})

        @dataclass
        class _Result:
            question: str
            answer: str
            trajectory: object = None

        return _Result(question=question, answer=f"answer {self.turn}")


def _run_three(tmp_path: Path, m_level: str):
    dialog = Dialog(
        id="7",
        turns=[
            DialogTurn(n=1, text="first"),
            DialogTurn(n=2, text="second"),
            DialogTurn(n=3, text="third"),
        ],
    )
    handed: list[Path | None] = []

    def factory(turn: int, workspace: Path, turns_dir: Path | None):
        handed.append(turns_dir)
        return _FakeRunner(turn, workspace, turns_dir)

    result = asyncio.run(
        run_dialog(
            dialog,
            dialog_root=tmp_path / "dlg",
            runner_factory=factory,
            m_level=m_level,
        )
    )
    return result, handed


def test_each_turn_gets_its_own_workspace(tmp_path: Path) -> None:
    result, _ = _run_three(tmp_path, "m1")

    assert [t.workspace.name for t in result.turns] == [
        "turn-01",
        "turn-02",
        "turn-03",
    ]
    assert (tmp_path / "dlg" / "turn-02" / "out-2.txt").is_file()


def test_m1_hands_every_turn_after_the_first_a_staged_tree(tmp_path: Path) -> None:
    _, handed = _run_three(tmp_path, "m1")

    assert handed[0] is None, "turn 1 has no history to mount"
    assert handed[1] is not None
    assert handed[2] is not None
    assert (handed[2] / "turn-02" / "ANSWER.md").read_text().strip() == "answer 2"


def test_m0_hands_nothing_to_any_turn(tmp_path: Path) -> None:
    _, handed = _run_three(tmp_path, "m0")

    assert handed == [None, None, None]


def test_the_mounted_tree_grows_by_one_turn_each_time(tmp_path: Path) -> None:
    _, handed = _run_three(tmp_path, "m1")

    assert sorted(p.name for p in handed[1].iterdir()) == ["turn-01", "turn-router"]
    assert sorted(p.name for p in handed[2].iterdir()) == [
        "turn-01",
        "turn-02",
        "turn-router",
    ]


def test_dialog_json_records_the_arm_and_every_turn(tmp_path: Path) -> None:
    result, _ = _run_three(tmp_path, "m1")

    written = json.loads((tmp_path / "dlg" / "dialog.json").read_text())
    assert written["m_level"] == "m1"
    assert written["turn_count"] == 3
    assert [t["n"] for t in written["turns"]] == [1, 2, 3]
    assert all(t["duration_ms"] >= 0 for t in written["turns"])
    assert result.turns[1].answer == "answer 2"


def test_a_failed_turn_does_not_end_the_dialog(tmp_path: Path) -> None:
    """The paper's recovery metric needs the dialog to continue past a failure."""
    dialog = Dialog(
        id="7",
        turns=[DialogTurn(n=1, text="boom"), DialogTurn(n=2, text="carry on")],
    )

    class _Exploding(_FakeRunner):
        async def run(self, question: str):
            if self.turn == 1:
                raise RuntimeError("tool exploded")
            return await super().run(question)

    staged: list[Path | None] = []

    def factory(turn: int, workspace: Path, turns_dir: Path | None):
        staged.append(turns_dir)
        return _Exploding(turn, workspace, turns_dir)

    result = asyncio.run(
        run_dialog(
            dialog,
            dialog_root=tmp_path / "dlg",
            runner_factory=factory,
            m_level="m1",
        )
    )

    assert result.turns[0].failed is True
    assert "tool exploded" in result.turns[0].answer
    assert result.turns[1].failed is False
    # And the failure is visible to turn 2, which is the point.
    assert "(failed)" in (staged[1] / "turn-router" / "SKILL.md").read_text()
