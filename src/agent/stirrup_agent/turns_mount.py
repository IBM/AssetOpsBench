"""Turn mounting for multi-turn dialogs (the memory plug).

Stirrup has no multi-turn dialog mechanism. ``Agent.run()`` rebuilds the
conversation from scratch on every call: it appends a fresh ``SystemMessage``
and resets ``full_msg_history`` to ``[]``, so nothing carries from one
``run()`` to the next. The only restore path is ``resume=True``, keyed by
``compute_task_hash(init_msgs)``, which resumes an interrupted run of the *same*
task rather than continuing a conversation.

This module adds the smallest thing that makes a dialog usable there, and it
deliberately mirrors ``skills_mount``: a tree is copied into the
code-execution workspace and one short block naming a router is appended to the
system prompt. The agent already has a shell, so progressive disclosure comes
free. Router, then one turn, then that turn's files.

Why the filesystem rather than the message history
--------------------------------------------------
Replaying prior messages as ``init_msgs`` is the obvious alternative and it is
worse in three ways. ``Agent._get_turn_count`` counts ``AssistantMessage``
instances across the history, and the loop guard is
``while _get_turn_count(...) < max_turns``, so every replayed assistant message
spends the agent's working budget before it does any new work. Replay also puts
the whole prior trace in the prompt whether or not the turn needs it, which is
the context growth the baseline in the dialog paper suffers from. And replay is
invisible: you cannot tell from the trajectory whether the agent used turn 2's
evidence or ignored it.

Mounting instead makes retrieval an action. The agent reads the router, decides
which earlier turn matters, and opens it. That decision lands in the trajectory
as a ``code_exec`` call naming a path, so cross-turn reuse becomes something you
measure rather than something you assume.

``M_LEVEL`` is the dialog control, exactly as ``K_LEVEL`` is the skills control.
``m0`` mounts nothing, so each turn runs as if it were the first and you measure
how much the dialog actually depends on its history. ``m1`` mounts the routed
tree. ``m1-full`` mounts the same tree without the routing discipline, which
isolates routing from mere availability.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .skills_mount import copy_tree_into

_log = logging.getLogger(__name__)

M_LEVELS = ("m0", "m1", "m1-full")

MOUNT_NAME = "turns"
ROUTER_DIRNAME = "turn-router"

_TURNS_PROMPT = """\
This is turn {turn} of a dialog. Earlier turns of the same dialog are mounted at
{mount}. They hold what the user asked before, what you answered, and the files
you produced while answering.

Route before you act. Read {mount}/{router}/SKILL.md first. It lists each
earlier turn, what was asked, and where that turn's files are. Open one turn's
ANSWER.md when the router says it bears on the current question, and its
workspace/ only when you need the artifact itself. Do not read every turn.

The user speaks as if you remember. When this turn says "that chiller", "the
same anomaly" or "the one you found", resolve the reference from the router
before you re-derive it. Evidence you already gathered is on disk: reuse it
rather than calling the same tool again.
"""

_FULL_PROMPT = """\
This is turn {turn} of a dialog. Earlier turns of the same dialog are mounted at
{mount}, including what was asked, what you answered, and the files produced.

The user speaks as if you remember. Resolve references to earlier turns from
what is mounted there.
"""

_ROUTER_HEADER = """\
---
name: turn-router
description: Index of earlier turns in this dialog. Read this first, then open \
only the turn that bears on the current question.
leakage-class: trajectory
---

# Earlier turns in this dialog

Turn {current} is the one you are answering now. Everything below already
happened. Each turn's directory holds:

- `ASK.md` - what the user asked on that turn, verbatim.
- `ANSWER.md` - the answer you gave.
- `workspace/` - the files that turn left behind, if any.

Open the row you need. Do not read every turn.

"""

_ROUTER_FOOTER = """

## Using this

Resolve a pronoun or a definite reference ("that chiller", "the same window")
against the Asked column before you re-derive anything. When a row's answer
already contains what this turn needs, cite it rather than calling the tool
again. When a turn produced a file, its path under `workspace/` is the artifact
itself, and reading it costs one shell command.

A turn that failed is still evidence. It tells you which approach not to repeat.
"""


@dataclass
class TurnRecord:
    """One completed turn, as the next turn gets to see it."""

    n: int
    ask: str
    answer: str
    workspace: Path | None = None
    tool_calls: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    failed: bool = False

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "ask": self.ask,
            "answer": self.answer,
            "workspace": str(self.workspace) if self.workspace else None,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
            "failed": self.failed,
        }


def resolve_m_level(m_level: str, turn: int) -> bool:
    """Whether turn ``turn`` should carry a mount. Raises on a bad level."""
    if m_level not in M_LEVELS:
        raise ValueError(f"m_level must be one of {M_LEVELS}, got {m_level!r}")
    if m_level == "m0":
        return False
    return turn > 1


def _summarize(text: str, limit: int = 160) -> str:
    """One line for the router table, with pipes escaped."""
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + "…"
    return flat.replace("|", "\\|")


def build_router(records: list[TurnRecord], current_turn: int) -> str:
    """Render the router index over completed turns."""
    lines = [_ROUTER_HEADER.format(current=current_turn)]
    lines.append("| Turn | Asked | Answered | Files |")
    lines.append("| --- | --- | --- | --- |")
    for record in records:
        directory = f"turn-{record.n:02d}"
        files = f"`{directory}/workspace/`" if record.workspace else "none"
        status = " (failed)" if record.failed else ""
        lines.append(
            f"| {record.n} | {_summarize(record.ask)} | "
            f"`{directory}/ANSWER.md`{status} | {files} |"
        )
    lines.append(_ROUTER_FOOTER)

    for record in records:
        lines.append(f"\n## Turn {record.n}\n")
        lines.append(f"Asked: {_summarize(record.ask, 400)}\n")
        if record.tool_calls:
            unique = sorted(set(record.tool_calls))
            lines.append(f"Tools used: {', '.join(unique)}\n")
        lines.append(f"Answer: `turn-{record.n:02d}/ANSWER.md`\n")
    return "\n".join(lines)


def stage_turns(
    records: list[TurnRecord], staging_dir: Path | str, current_turn: int
) -> Path:
    """Assemble the mountable tree for the turns completed so far.

    Layout, which is what the router promises the agent::

        <staging>/turn-router/SKILL.md
        <staging>/turn-01/ASK.md
        <staging>/turn-01/ANSWER.md
        <staging>/turn-01/workspace/...
    """
    staging = Path(staging_dir).expanduser().resolve()
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    router_dir = staging / ROUTER_DIRNAME
    router_dir.mkdir()
    (router_dir / "SKILL.md").write_text(
        build_router(records, current_turn), encoding="utf-8"
    )

    for record in records:
        turn_dir = staging / f"turn-{record.n:02d}"
        turn_dir.mkdir()
        (turn_dir / "ASK.md").write_text(record.ask + "\n", encoding="utf-8")
        (turn_dir / "ANSWER.md").write_text(record.answer + "\n", encoding="utf-8")
        (turn_dir / "turn.json").write_text(
            json.dumps(record.to_json(), indent=2), encoding="utf-8"
        )
        if record.workspace is not None and Path(record.workspace).is_dir():
            _copy_workspace(Path(record.workspace), turn_dir / "workspace")

    _log.info("staged %d earlier turns at %s", len(records), staging)
    return staging


def _copy_workspace(source: Path, destination: Path) -> None:
    """Copy a preserved turn workspace, minus anything we mounted into it.

    A preserved workspace contains whatever the previous turn's exec dir held,
    which includes the trees we mounted for that turn. Carrying those forward
    would nest turn N-1's copy of turn N-2 inside turn N's copy of turn N-1, and
    the tree would grow quadratically down the dialog.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {MOUNT_NAME, "skills"}:
            continue
        if item.is_dir():
            shutil.copytree(item, destination / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination / item.name)


def turns_prompt(
    turn: int, m_level: str = "m1", code_backend: str = "docker"
) -> str | None:
    """Return the system-prompt block for the turn mount, or None."""
    if not resolve_m_level(m_level, turn):
        return None
    mount = f"/workspace/{MOUNT_NAME}" if code_backend == "docker" else MOUNT_NAME
    template = _FULL_PROMPT if m_level == "m1-full" else _TURNS_PROMPT
    return template.format(turn=turn, mount=mount, router=ROUTER_DIRNAME)


def copy_turns_into(staging_dir: Path | str, exec_dir: Path | str) -> int:
    """Mount the staged turns at ``<exec dir>/turns``."""
    return copy_tree_into(staging_dir, exec_dir, name=MOUNT_NAME)
