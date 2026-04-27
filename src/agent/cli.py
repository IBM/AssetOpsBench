"""CLI entry point for the plan-execute runner.

Usage:
    plan-execute "What assets are available at site MAIN?"
    plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "List sensors"
    plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "What are the failure modes?"
    plan-execute --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import json
import sys

from ._cli_common import HR, add_common_args, run_sdk_cli

_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plan-execute",
        description="Run a question through the MCP plan-execute workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  The provider is encoded in the model-id prefix:
    watsonx/<model>          IBM WatsonX  (e.g. watsonx/meta-llama/llama-3-3-70b-instruct)
    litellm_proxy/<model>    LiteLLM proxy (e.g. litellm_proxy/GCP/claude-4-sonnet)

environment variables:
  WATSONX_APIKEY        IBM WatsonX API key      (required for watsonx/* models)
  WATSONX_PROJECT_ID    IBM WatsonX project ID   (required for watsonx/* models)
  WATSONX_URL           IBM WatsonX endpoint     (optional, defaults to us-south)

  LITELLM_API_KEY       LiteLLM API key          (required for non-watsonx models)
  LITELLM_BASE_URL      LiteLLM base URL         (required for non-watsonx models)

  LOG_LEVEL             Log level for MCP servers (default: WARNING)

examples:
  plan-execute "What assets are at site MAIN?"
  plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "List sensors"
  plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "What are the failure modes?"
  plan-execute --verbose --show-trajectory --json "How many IoT observations exist for CH-1?"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated plan before execution.",
    )
    return parser


def _build_llm(model_id: str):
    try:
        from llm.litellm import LiteLLMBackend
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        return LiteLLMBackend(model_id=model_id)
    except KeyError as exc:
        print(f"error: missing environment variable {exc}", file=sys.stderr)
        sys.exit(1)


def _print_section(title: str) -> None:
    print(f"\n{HR}")
    print(f"  {title}")
    print(HR)


async def _run(args: argparse.Namespace) -> None:
    from agent.plan_execute.runner import PlanExecuteRunner

    llm = _build_llm(args.model_id)
    runner = PlanExecuteRunner(llm=llm)
    result = await runner.run(args.question)

    if args.output_json:
        output = {
            "question": result.question,
            "answer": result.answer,
            "plan": [
                {
                    "step": s.step_number,
                    "task": s.task,
                    "server": s.server,
                    "tool": s.tool,
                    "tool_args": s.tool_args,
                    "dependencies": s.dependencies,
                    "expected_output": s.expected_output,
                }
                for s in result.plan.steps
            ],
            "trajectory": [
                {
                    "step": r.step_number,
                    "task": r.task,
                    "server": r.server,
                    "tool": r.tool,
                    "tool_args": r.tool_args,
                    "response": r.response,
                    "error": r.error,
                    "success": r.success,
                }
                for r in result.trajectory
            ],
        }
        print(json.dumps(output, indent=2))
        return

    if args.show_plan:
        _print_section("Plan")
        for step in result.plan.steps:
            deps = ", ".join(f"#{d}" for d in step.dependencies) or "none"
            print(f"  [{step.step_number}] {step.server}: {step.task}")
            print(f"       tool: {step.tool}  args: {step.tool_args}")
            print(f"       deps={deps} | expected: {step.expected_output}")

    if args.show_trajectory:
        _print_section("Trajectory")
        for r in result.trajectory:
            status = "OK " if r.success else "ERR"
            print(f"  [{status}] Step {r.step_number} ({r.server}): {r.task}")
            if r.tool and r.tool.lower() not in ("none", "null", ""):
                print(f"       tool: {r.tool}  args: {r.tool_args}")
            detail = r.response if r.success else f"Error: {r.error}"
            snippet = detail[:200] + ("..." if len(detail) > 200 else "")
            print(f"        {snippet}")

    _print_section("Answer")
    print(result.answer)
    print()


def main() -> None:
    run_sdk_cli("plan-execute", _build_parser, _run)


if __name__ == "__main__":
    main()
