"""The outer loop that turns a Stirrup single-shot runner into a dialog.

Stirrup gives one run, not a conversation. This module wraps it:

    for each turn:
        stage the turns completed so far into a mountable tree
        run the turn in its own workspace, with that tree mounted
        preserve the workspace, and record ask / answer / files

Each turn gets its own directory under the dialog root::

    <dialog-root>/
      turn-01/            --workspace-dir for turn 1, preserved after it
      turn-02/            turn 2, with turn 1 mounted at /workspace/turns
      _staged/turn-02/    the tree mounted into turn 2
      dialog.json         the record of the whole dialog

Turn N never sees turn N's own directory as history; it sees the staged tree
built from turns 1..N-1. The staging step is what keeps the mount honest: the
agent reads a router and chooses, rather than inheriting a directory.

Why a fresh session per turn
----------------------------
Holding one Stirrup session open for the whole dialog would keep the same
``temp_dir`` across turns, and continuity would come free. It would also make
cross-turn reuse unobservable and untestable: no mount, no routing decision, no
way to run an ``m0`` arm. A session per turn costs a container start and buys a
controlled experiment, which is the trade this benchmark exists to make.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import AgentResult
from .turns_mount import TurnRecord, stage_turns

_log = logging.getLogger(__name__)


@dataclass
class DialogTurn:
    """One authored turn of a dialog."""

    n: int
    text: str
    characteristic_form: str | None = None
    expected_answer: str | None = None
    depends_on: list[int] = field(default_factory=list)


@dataclass
class Dialog:
    """An authored multi-turn scenario."""

    id: str
    turns: list[DialogTurn]
    type: str = ""
    category: str = ""

    @property
    def is_single_turn(self) -> bool:
        return len(self.turns) == 1


@dataclass
class TurnResult:
    """What one executed turn produced."""

    n: int
    ask: str
    answer: str
    workspace: Path
    duration_ms: float
    tool_calls: list[str]
    failed: bool
    result: AgentResult | None = None

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "ask": self.ask,
            "answer": self.answer,
            "workspace": str(self.workspace),
            "duration_ms": round(self.duration_ms, 1),
            "tool_calls": self.tool_calls,
            "failed": self.failed,
        }


@dataclass
class DialogResult:
    """Every turn of one dialog, in order."""

    dialog_id: str
    m_level: str
    k_level: str
    turns: list[TurnResult]

    def to_json(self) -> dict:
        return {
            "dialog_id": self.dialog_id,
            "m_level": self.m_level,
            "k_level": self.k_level,
            "turn_count": len(self.turns),
            "turns": [t.to_json() for t in self.turns],
        }


def load_dialog(scenario_dir: Path | str) -> Dialog:
    """Read a dialog from a scenario directory.

    The layout extends the existing one rather than replacing it::

        scenario_7/
          question.txt          turn 1, exactly as today
          groundtruth.txt       turn 1's expected answer, exactly as today
          turns/
            02/question.txt     turn 2
            02/groundtruth.txt
            03/question.txt

    A scenario with no ``turns/`` directory loads as a one-turn dialog, so every
    scenario in the suite is already a valid dialog and nothing needs editing.
    """
    root = Path(scenario_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"scenario directory not found: {root}")

    question_path = root / "question.txt"
    if not question_path.is_file():
        raise ValueError(f"no question.txt in {root}")

    scenario_id = root.name.removeprefix("scenario_")
    turns = [
        DialogTurn(
            n=1,
            text=question_path.read_text(encoding="utf-8").strip(),
            expected_answer=_read_optional(root / "groundtruth.txt"),
            characteristic_form=_read_optional(root / "characteristic_form.txt"),
        )
    ]

    turns_root = root / "turns"
    if turns_root.is_dir():
        for turn_dir in sorted(p for p in turns_root.iterdir() if p.is_dir()):
            try:
                n = int(turn_dir.name)
            except ValueError as exc:
                raise ValueError(
                    f"turn directory must be a number, got {turn_dir.name!r} "
                    f"in {turns_root}"
                ) from exc
            if n < 2:
                raise ValueError(
                    f"turn directories start at 02 (turn 1 is question.txt); "
                    f"got {turn_dir.name!r}"
                )
            turn_question = turn_dir / "question.txt"
            if not turn_question.is_file():
                raise ValueError(f"no question.txt in {turn_dir}")
            turns.append(
                DialogTurn(
                    n=n,
                    text=turn_question.read_text(encoding="utf-8").strip(),
                    expected_answer=_read_optional(turn_dir / "groundtruth.txt"),
                    characteristic_form=_read_optional(
                        turn_dir / "characteristic_form.txt"
                    ),
                    depends_on=_read_depends_on(turn_dir / "depends_on.txt"),
                )
            )

    expected = list(range(1, len(turns) + 1))
    actual = [t.n for t in turns]
    if actual != expected:
        raise ValueError(
            f"dialog {scenario_id} has turns {actual}, expected {expected}; "
            "turn numbers must be contiguous from 1"
        )
    return Dialog(id=scenario_id, turns=turns)


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _read_depends_on(path: Path) -> list[int]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8").replace(",", " ").split()
    return [int(token) for token in raw]


async def run_dialog(
    dialog: Dialog,
    *,
    dialog_root: Path | str,
    runner_factory,
    m_level: str = "m1",
    k_level: str = "k0",
) -> DialogResult:
    """Run every turn, mounting the turns completed so far into the next.

    ``runner_factory(turn_number, workspace_dir, turns_dir)`` returns a
    configured ``StirrupAgentRunner``. Injecting it keeps this loop free of
    model, backend and skill wiring, and lets the tests drive it with a fake.
    """
    root = Path(dialog_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging_root = root / "_staged"

    records: list[TurnRecord] = []
    results: list[TurnResult] = []

    for turn in dialog.turns:
        workspace = root / f"turn-{turn.n:02d}"
        workspace.mkdir(parents=True, exist_ok=True)

        turns_dir: Path | None = None
        if records and m_level != "m0":
            turns_dir = stage_turns(
                records, staging_root / f"turn-{turn.n:02d}", current_turn=turn.n
            )

        runner = runner_factory(turn.n, workspace, turns_dir)

        _log.info(
            "dialog %s turn %d/%d (m_level=%s, mounted=%s)",
            dialog.id,
            turn.n,
            len(dialog.turns),
            m_level,
            turns_dir is not None,
        )

        started = time.perf_counter()
        failed = False
        answer = ""
        result = None
        try:
            result = await runner.run(turn.text)
            answer = result.answer
        except Exception as exc:  # a failed turn is still evidence
            failed = True
            answer = f"Turn failed: {type(exc).__name__}: {exc}"
            _log.warning("dialog %s turn %d failed", dialog.id, turn.n, exc_info=True)
        duration_ms = (time.perf_counter() - started) * 1000

        tool_calls = _tool_names(result)
        results.append(
            TurnResult(
                n=turn.n,
                ask=turn.text,
                answer=answer,
                workspace=workspace,
                duration_ms=duration_ms,
                tool_calls=tool_calls,
                failed=failed,
                result=result,
            )
        )
        records.append(
            TurnRecord(
                n=turn.n,
                ask=turn.text,
                answer=answer,
                workspace=workspace,
                tool_calls=tool_calls,
                duration_ms=duration_ms,
                failed=failed,
            )
        )

    dialog_result = DialogResult(
        dialog_id=dialog.id, m_level=m_level, k_level=k_level, turns=results
    )
    (root / "dialog.json").write_text(
        json.dumps(dialog_result.to_json(), indent=2), encoding="utf-8"
    )
    return dialog_result


def _tool_names(result: AgentResult | None) -> list[str]:
    if result is None or result.trajectory is None:
        return []
    try:
        return [tc.name for tc in result.trajectory.all_tool_calls]
    except AttributeError:
        return []
