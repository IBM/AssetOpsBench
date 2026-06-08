#!/usr/bin/env python
"""Quick smoke test: run the Claude agent over the FMC work-order scenarios.

Loads the scenarios from ``src/scenarios/local/workorder_utterance.json``, runs
each one through ``ClaudeAgentRunner`` (which connects the ``wo`` MCP server and
its FMC tools), and prints the agent's answer next to the expected answer plus
the tools it actually called.

Write-back scenarios (S2/S4/S5) mutate CouchDB; by default this script
re-blanks every ``TST-`` record afterwards so the evaluation dataset stays
pristine. Pass ``--no-restore`` to leave the imputations in place (e.g. to
inspect the write-back independently).

Requires CouchDB up (``workorder`` DB loaded) and LITELLM_* env vars in .env.

Usage:
    uv run python scripts/test_fmc_claude_agent.py                 # all scenarios
    uv run python scripts/test_fmc_claude_agent.py S1 S3           # only S1 and S3
    uv run python scripts/test_fmc_claude_agent.py --no-restore
    uv run python scripts/test_fmc_claude_agent.py --show-trajectory
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

_SCENARIOS = _SRC / "scenarios" / "local" / "workorder_utterance.json"
_HR = "=" * 72


def _load_scenarios(labels: set[str]) -> list[dict]:
    import json

    data = json.loads(_SCENARIOS.read_text(encoding="utf-8"))
    out = []
    for s in data:
        label = s.get("metadata", {}).get("scenario_label", "")
        if labels and label not in labels:
            continue
        out.append(s)
    return out


async def _run(args: argparse.Namespace) -> None:
    from agent.claude_agent.runner import ClaudeAgentRunner
    from observability import set_run_context
    from servers.wo.data import load, write_failure_codes

    scenarios = _load_scenarios(set(args.labels))
    if not scenarios:
        print(f"No scenarios matched {args.labels!r}")
        return

    traj_dir = os.environ.get("AGENT_TRAJECTORY_DIR")
    if traj_dir:
        print(f"[trajectory] saving per-scenario JSON to {traj_dir}")

    runner = ClaudeAgentRunner(model=args.model_id, max_turns=args.max_turns)
    needs_restore = False

    for s in scenarios:
        md = s["metadata"]
        label = md.get("scenario_label", "?")
        needs_restore = needs_restore or bool(md.get("write_back"))

        # Fresh run_id per scenario so persist_trajectory writes one file each
        # (keyed by run_id, with scenario_id recorded inside).
        run_id = str(uuid.uuid4())
        set_run_context(run_id=run_id, scenario_id=label)

        print(f"\n{_HR}\n{label} · {md.get('subtitle', '')}\n{_HR}")
        print(f"Q: {s['text']}\n")

        result = await runner.run(s["text"])

        tools_used = [tc.name for tc in result.trajectory.all_tool_calls]
        print(f"EXPECTED      : {s.get('expected_answer')}")
        print(f"AGENT ANSWER  : {result.answer}")
        print(f"EXPECTED TOOLS: {md.get('expected_tools')}")
        print(f"TOOLS USED    : {tools_used}")
        print(
            f"(turns={len(result.trajectory.turns)}, "
            f"tool_calls={len(tools_used)}, "
            f"out_tokens={result.trajectory.total_output_tokens})"
        )
        if traj_dir:
            print(f"SAVED         : {Path(traj_dir) / f'{run_id}.json'}  (scenario_id={label})")
        if args.show_trajectory:
            for t in result.trajectory.turns:
                for tc in t.tool_calls:
                    print(f"    → {tc.name}({tc.input})")

    if needs_restore and not args.no_restore:
        df = load("wo_fmc")
        status = {}
        if df is not None:
            tst_ids = [str(w) for w in df.loc[df["wo_id"].str.startswith("TST"), "wo_id"]]
            status = write_failure_codes({wo_id: None for wo_id in tst_ids}) or {}
        restored = sum(1 for ok in status.values() if ok)
        print(f"\n[restore] re-blanked {restored} TST- record(s) to keep the dataset pristine.")
    elif needs_restore:
        print("\n[restore] skipped (--no-restore); TST- records keep their imputed codes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="*", help="Scenario labels to run (e.g. S1 S3). Default: all.")
    parser.add_argument("--model-id", default="litellm_proxy/aws/claude-opus-4-6", help="Model string.")
    parser.add_argument("--max-turns", type=int, default=30, help="Max agentic loop turns.")
    parser.add_argument("--no-restore", action="store_true", help="Leave write-back imputations in CouchDB.")
    parser.add_argument("--show-trajectory", action="store_true", help="Print each tool call.")
    parser.add_argument(
        "--trajectory-dir",
        metavar="DIR",
        default=None,
        help="Save a {run_id}.json trajectory per scenario to DIR "
        "(overrides/sets AGENT_TRAJECTORY_DIR).",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    if args.trajectory_dir:
        os.environ["AGENT_TRAJECTORY_DIR"] = args.trajectory_dir
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
