"""Run benchmark MCP tools using Codex subscription authentication."""
import argparse
from .._cli_common import add_common_args, print_result, run_sdk_cli


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_model="gpt-6-astra")
    return parser


async def _run(args):
    from .runner import CodexAgentRunner
    result = await CodexAgentRunner(model=args.model_id).run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main():
    run_sdk_cli("codex-agent", _parser, _run)


if __name__ == "__main__":
    main()
