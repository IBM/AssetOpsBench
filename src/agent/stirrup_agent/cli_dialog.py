"""CLI entry point for running a multi-turn dialog through the Stirrup agent.

Usage:
    stirrup-dialog --scenario-dir src/couchdb/scenarios_data/scenario_1 \\
      --dialog-root ./dlg-1 --m-level m1

    # the m0 arm: every turn runs as if it were the first
    stirrup-dialog --scenario-dir ... --dialog-root ./dlg-1-m0 --m-level m0

    # an ad-hoc dialog, no scenario directory needed
    stirrup-dialog --dialog-root ./dlg-ad-hoc \\
      --turn "What sensors are on Chiller 6?" \\
      --turn "Which of those has drifted this month?" \\
      --turn "Raise a work order for the worst one."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .._cli_common import add_common_args, run_sdk_cli

_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stirrup-dialog",
        description=(
            "Run a multi-turn dialog through the Stirrup agent. Each turn runs "
            "in its own workspace; earlier turns are mounted into the next one "
            "and reached through a router, the same way skills are."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
m-level (the dialog control, alongside --k-level for skills):
  m0        Mount nothing. Every turn runs as if it were the first. The
            unaided arm: it measures how much the dialog needs its history.
  m1        Mount earlier turns behind a router (default).
  m1-full   Mount the same tree with no routing discipline, which separates
            routing from mere availability.

dialog layout on disk:
  scenario_7/
    question.txt          turn 1, exactly as today
    groundtruth.txt       turn 1's expected answer
    turns/02/question.txt turn 2
    turns/03/question.txt turn 3

  A scenario with no turns/ directory is a one-turn dialog, so the whole
  existing suite already loads.
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL, include_question=False)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scenario-dir",
        type=Path,
        metavar="PATH",
        help="Scenario directory holding question.txt and an optional turns/.",
    )
    source.add_argument(
        "--turn",
        action="append",
        dest="turns",
        metavar="TEXT",
        help="One turn of an ad-hoc dialog. Repeat, in order.",
    )

    parser.add_argument(
        "--dialog-root",
        type=Path,
        required=True,
        metavar="PATH",
        help="Directory for per-turn workspaces, staged mounts and dialog.json.",
    )
    parser.add_argument(
        "--m-level",
        choices=("m0", "m1", "m1-full"),
        default="m1",
        help="Dialog memory level (default: m1).",
    )
    parser.add_argument(
        "--k-level",
        choices=("k0", "k1", "k1-recovery"),
        default="k0",
        help="Skill level, passed through to every turn (default: k0).",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Skill collection, required when --k-level is not k0.",
    )
    parser.add_argument(
        "--code-backend",
        choices=["docker", "local"],
        default="docker",
        help="Code-execution sandbox backend (default: docker).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        metavar="N",
        help="Stirrup agent-loop bound, applied per dialog turn (default: 30).",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.stirrup_agent.dialog import Dialog, DialogTurn, load_dialog, run_dialog
    from agent.stirrup_agent.runner import StirrupAgentRunner

    if args.scenario_dir is not None:
        dialog = load_dialog(args.scenario_dir)
    else:
        dialog = Dialog(
            id="ad-hoc",
            turns=[DialogTurn(n=i, text=t) for i, t in enumerate(args.turns, start=1)],
        )

    def runner_factory(turn: int, workspace: Path, turns_dir: Path | None):
        return StirrupAgentRunner(
            model=args.model_id,
            code_enabled=True,
            code_backend=args.code_backend,
            workspace_dir=workspace,
            preserve_workspace=True,
            skills_dir=args.skills_dir,
            k_level=args.k_level,
            turns_dir=turns_dir,
            m_level=args.m_level,
            turn=turn,
            max_turns=args.max_turns,
        )

    result = await run_dialog(
        dialog,
        dialog_root=args.dialog_root,
        runner_factory=runner_factory,
        m_level=args.m_level,
        k_level=args.k_level,
    )

    if args.output_json:
        print(json.dumps(result.to_json(), indent=2))
        return

    print(f"\nDialog {result.dialog_id}  m_level={result.m_level}  "
          f"k_level={result.k_level}  turns={len(result.turns)}\n")
    for turn in result.turns:
        status = "FAILED" if turn.failed else "ok"
        print(f"--- Turn {turn.n} [{status}] {turn.duration_ms / 1000:.1f}s "
              f"({len(turn.tool_calls)} tool calls)")
        print(f"Q: {turn.ask}")
        print(f"A: {turn.answer}\n")
    print(f"Record written to {Path(args.dialog_root) / 'dialog.json'}")


def main() -> None:
    run_sdk_cli("stirrup-dialog", _build_parser, _run)


if __name__ == "__main__":
    main()
