"""Tests for the Stirrup agent CLI arguments."""

from agent.stirrup_agent.cli import _build_parser


def test_cli_accepts_max_reasoning_effort() -> None:
    args = _build_parser().parse_args(
        ["Return OK.", "--reasoning-effort", "max"]
    )

    assert args.reasoning_effort == "max"
