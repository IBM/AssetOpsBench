"""CLI for ReActXen over the Stirrup core."""

from .._cli_common import print_result, run_sdk_cli
from ..stirrup_agent.cli import _build_parser as stirrup_parser


def _build_parser():
    parser = stirrup_parser()
    parser.prog = "reactxen-agent"
    parser.description = "Run ReActXen with Stirrup, self-ask, review and reflection."
    parser.add_argument("--max-rounds", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument(
        "--critic-mode",
        choices=["react", "completion"],
        default="react",
        help="Workspace-only Stirrup critics (default), or legacy completion critics.",
    )
    parser.add_argument("--review-max-turns", type=int, default=10)
    parser.add_argument("--reflect-max-turns", type=int, default=8)
    parser.add_argument(
        "--review-finish-repair-attempts", type=int, choices=range(4), default=2
    )
    parser.add_argument(
        "--reflect-finish-repair-attempts", type=int, choices=range(4), default=2
    )
    distill = parser.add_mutually_exclusive_group()
    distill.add_argument(
        "--distill-question", dest="distill_question", action="store_true", default=True
    )
    distill.add_argument(
        "--no-distill-question", dest="distill_question", action="store_false"
    )
    parser.add_argument("--no-self-ask", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    return parser


async def _run(args):
    from .runner import ReActXenAgentRunner

    runner = ReActXenAgentRunner(
        model=args.model_id,
        critic_model=args.critic_model,
        critic_mode=args.critic_mode,
        review_max_turns=args.review_max_turns,
        reflect_max_turns=args.reflect_max_turns,
        reflect_finish_repair_attempts=args.reflect_finish_repair_attempts,
        review_finish_repair_attempts=args.review_finish_repair_attempts,
        distill_question_enabled=args.distill_question,
        max_rounds=args.max_rounds,
        self_ask_enabled=not args.no_self_ask,
        review_enabled=not args.no_review,
        code_enabled=args.code_enabled,
        code_backend=args.code_backend,
        workspace_dir=args.workspace_dir,
        preserve_workspace=args.preserve_workspace,
        max_turns=args.max_turns,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    result = await runner.run(args.question)
    print_result(
        result, show_trajectory=args.show_trajectory, output_json=args.output_json
    )
    if not result.completed:
        raise SystemExit(2)


def main():
    run_sdk_cli("reactxen-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
