#!/usr/bin/env python3
"""Run all scenarios from a utterance JSON file and report average latency per stage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCENARIOS_FILE = Path(__file__).parent.parent / "src/scenarios/local/vibration_utterance.json"
DEFAULT_MODEL = "openai/Qwen/Qwen2.5-7B-Instruct"


def run_scenario(text: str, model_id: str) -> dict | None:
    result = subprocess.run(
        ["uv", "run", "plan-execute", "--model-id", model_id, "--json", text],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse JSON: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark plan-execute latency over a scenario set.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--scenarios", default=str(SCENARIOS_FILE), help="Path to utterance JSON file")
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    scenarios = json.loads(scenarios_path.read_text())
    print(f"Model:     {args.model_id}")
    print(f"Scenarios: {scenarios_path.name} ({len(scenarios)} total)\n")

    results = []
    for i, scenario in enumerate(scenarios, 1):
        text = scenario["text"]
        sid = scenario.get("id", i)
        print(f"[{i}/{len(scenarios)}] id={sid}: {text[:80]}")
        output = run_scenario(text, args.model_id)
        if output and "latency" in output:
            lat = output["latency"]
            results.append(lat)
            print(f"  plan={lat['plan']:.3f}s  execute={lat['execute']:.3f}s  "
                  f"summarize={lat['summarize']:.3f}s  total={lat['total']:.3f}s")
        else:
            print("  [SKIPPED] no latency data")

    if not results:
        print("\nNo results collected.")
        return

    n = len(results)
    avg = {
        "plan":      sum(r["plan"]      for r in results) / n,
        "execute":   sum(r["execute"]   for r in results) / n,
        "summarize": sum(r["summarize"] for r in results) / n,
        "total":     sum(r["total"]     for r in results) / n,
    }

    print(f"\n{'─' * 50}")
    print(f"  Average latency over {n} scenarios")
    print(f"{'─' * 50}")
    print(f"  Plan:       {avg['plan']:.3f}s  ({avg['plan']/avg['total']*100:.1f}%)")
    print(f"  Execute:    {avg['execute']:.3f}s  ({avg['execute']/avg['total']*100:.1f}%)")
    print(f"  Summarize:  {avg['summarize']:.3f}s  ({avg['summarize']/avg['total']*100:.1f}%)")
    print(f"  Total:      {avg['total']:.3f}s")
    print(f"{'─' * 50}")

    out_path = Path(__file__).parent / "latency_results.json"
    out_path.write_text(json.dumps({"summary": avg, "runs": results}, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
