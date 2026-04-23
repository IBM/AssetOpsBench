"""Shared CLI helpers for the SDK-driven agent entry points.

The three SDK CLIs (``claude-agent``, ``openai-agent``, ``deep-agent``) all
need to parse a question + model ID, set up logging, and print a
trajectory + answer section.  This module captures that common surface so
each CLI only has to encode its runner-specific differences
(prog name, default model, loop-bound arg like ``--max-turns`` vs
``--recursion-limit``, and which runner class to instantiate).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from typing import Awaitable, Callable

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
HR = "─" * 60


def setup_logging(verbose: bool) -> None:
    """Configure the root logger to stderr; INFO when verbose else WARNING."""
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def add_common_args(parser: argparse.ArgumentParser, default_model: str) -> None:
    """Register the args shared by every SDK CLI.

    Adds the positional ``question`` plus ``--model-id``, ``--show-trajectory``,
    ``--json``, and ``--verbose``.  The caller is responsible for any
    runner-specific flags (e.g. ``--max-turns``, ``--recursion-limit``).
    """
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--model-id",
        default=default_model,
        metavar="MODEL_ID",
        help=f"Model string (default: {default_model}).",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="Print each turn's text, tool calls, and token usage.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the full result as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level logs on stderr.",
    )


def print_trajectory(trajectory) -> None:
    """Pretty-print turns, tool calls, and token totals to stdout."""
    print(f"\n{HR}")
    print("  Trace")
    print(HR)
    for turn in trajectory.turns:
        print(
            f"\n  [Turn {turn.index}]  "
            f"in={turn.input_tokens} out={turn.output_tokens} tokens"
        )
        if turn.text:
            snippet = turn.text[:200] + ("..." if len(turn.text) > 200 else "")
            print(f"    text: {snippet}")
        for tc in turn.tool_calls:
            print(f"    tool: {tc.name}  input: {tc.input}")
            if tc.output is not None:
                out_str = str(tc.output)
                snippet = out_str[:200] + ("..." if len(out_str) > 200 else "")
                print(f"    output: {snippet}")
    print(
        f"\n  Total: {trajectory.total_input_tokens} input / "
        f"{trajectory.total_output_tokens} output tokens  "
        f"({len(trajectory.turns)} turns, "
        f"{len(trajectory.all_tool_calls)} tool calls)"
    )


def print_answer(answer: str) -> None:
    """Print the final answer section."""
    print(f"\n{HR}")
    print("  Answer")
    print(HR)
    print(answer)
    print()


def print_result(result, *, show_trajectory: bool, output_json: bool) -> None:
    """Common post-run output: JSON dump OR trajectory + answer sections."""
    if output_json:
        print(json.dumps(dataclasses.asdict(result.trajectory), indent=2, default=str))
        return
    if show_trajectory:
        print_trajectory(result.trajectory)
    print_answer(result.answer)


def run_sdk_cli(
    service_name: str,
    build_parser: Callable[[], argparse.ArgumentParser],
    run_coro: Callable[[argparse.Namespace], Awaitable[None]],
) -> None:
    """Run the standard SDK-CLI lifecycle.

    Loads ``.env``, parses args with the caller's parser factory, configures
    stderr logging, initialises OTEL tracing under *service_name*, and runs
    *run_coro* to completion via :func:`asyncio.run`.
    """
    from dotenv import load_dotenv

    from observability import init_tracing

    load_dotenv()
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    init_tracing(service_name)
    asyncio.run(run_coro(args))
