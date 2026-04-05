#!/usr/bin/env python3
"""Benchmark latency and accuracy for multi-agent scenarios 501-520.

Each scenario is run NUM_RUNS times:
  - Run 1: latency recorded + accuracy graded via LLM-as-judge
  - Runs 2-N: latency recorded only
  - Final latency is averaged across all runs

Usage:
    python scripts/bench_latency.py
    python scripts/bench_latency.py --model-id openai/Qwen/Qwen2.5-14B-Instruct
    python scripts/bench_latency.py --thinking
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "openai/Qwen/Qwen2.5-7B-Instruct"  # requires LITELLM_BASE_URL pointing to local vLLM
NUM_RUNS = 5

# ── LLM-as-judge prompt ───────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating an AI agent's response to an industrial asset operations query.

Question: {question}
Expected outcome: {characteristic_form}
Agent response: {answer}

Evaluate the response on the following six dimensions.

1. task_completion: Did the agent complete the requested task?
2. data_retrieval_accuracy: Was the correct data retrieved with correct parameters?
3. generalized_result_verification: Is the result consistent with general domain knowledge?
4. agent_sequence_correct: Were the steps executed in the correct logical order?
5. clarity_and_justification: Is the response clear and well-justified?
6. hallucinations: Did the agent hallucinate any facts not supported by the data?

Respond with a JSON object only, no explanation:
{{"task_completion": true, "data_retrieval_accuracy": true, "generalized_result_verification": true, "agent_sequence_correct": true, "clarity_and_justification": true, "hallucinations": false}}
"""


# ── dataset loading ───────────────────────────────────────────────────────────

def load_scenarios() -> list[dict]:
    """Load scenarios 501-520 from the HuggingFace AssetOpsBench dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package not installed. Run: pip3 install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset from HuggingFace...", flush=True)
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios", split="train")
    scenarios = [s for s in ds if 501 <= s["id"] <= 520]
    print(f"Loaded {len(scenarios)} scenarios (id 501-520)\n")
    return scenarios


# ── pipeline execution ────────────────────────────────────────────────────────

def run_scenario(text: str, model_id: str, thinking: bool = False) -> dict | None:
    cmd = ["uv", "run", "plan-execute", "--model-id", model_id, "--json"]
    if thinking:
        cmd.append("--thinking")
    cmd.append(text)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"    [ERROR] {result.stderr.strip()[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"    [ERROR] Failed to parse JSON: {e}", file=sys.stderr)
        return None


# ── grading ───────────────────────────────────────────────────────────────────

_judge_llm = None

def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        from llm.litellm import LiteLLMBackend
        _judge_llm = LiteLLMBackend(model_id="watsonx/meta-llama/llama-3-3-70b-instruct")
    return _judge_llm


def grade(question: str, characteristic_form: str, answer: str) -> dict:
    """Call LLM-as-judge and return per-dimension scores + overall pass/fail."""
    llm = _get_judge_llm()
    prompt = _JUDGE_PROMPT.format(
        question=question,
        characteristic_form=characteristic_form,
        answer=answer,
    )
    raw = llm.generate(prompt)

    text = raw.strip()
    if "```" in text:
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        scores = json.loads(text[start:end]) if start != -1 else {}
    except json.JSONDecodeError:
        scores = {}

    passed = (
        scores.get("task_completion", False)
        and scores.get("data_retrieval_accuracy", False)
        and scores.get("generalized_result_verification", False)
        and scores.get("agent_sequence_correct", False)
        and scores.get("clarity_and_justification", False)
        and not scores.get("hallucinations", True)
    )
    return {"scores": scores, "passed": passed}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark plan-execute on multi-agent scenarios 501-520.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help=f"Runs per scenario (default: {NUM_RUNS})")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode in the planning phase.")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Model:    {args.model_id}")
    print(f"Thinking: {'enabled' if args.thinking else 'disabled'}")
    print(f"Runs:     {args.runs} per scenario\n")

    all_results = []

    for i, scenario in enumerate(scenarios, 1):
        sid = scenario["id"]
        text = scenario["text"]
        characteristic_form = scenario.get("characteristic_form", "")
        print(f"[{i}/{len(scenarios)}] id={sid}: {text}")
        print(f"  expected: {characteristic_form}")

        latencies = []
        grade_result = None

        for run in range(1, args.runs + 1):
            print(f"  run {run}/{args.runs}", end=" ", flush=True)
            output = run_scenario(text, args.model_id, thinking=args.thinking)
            if output is None:
                print("[FAILED]")
                continue

            lat = output.get("latency", {})
            if lat:
                latencies.append(lat)
                print(f"total={lat['total']:.2f}s", end="")

            if run == 1 and grade_result is None and output.get("answer"):
                print(f"\n  answer: {output['answer']}")
                print(" grading...", end=" ", flush=True)
                grade_result = grade(text, characteristic_form, output["answer"])
                status = "PASS" if grade_result["passed"] else "FAIL"
                print(f"[{status}]")
            else:
                print()

        if not latencies:
            print("  [SKIPPED] no latency data\n")
            continue

        avg_lat = {k: sum(r[k] for r in latencies) / len(latencies)
                   for k in ("plan", "execute", "summarize", "total")}
        print(f"  avg: plan={avg_lat['plan']:.3f}s  execute={avg_lat['execute']:.3f}s  "
              f"summarize={avg_lat['summarize']:.3f}s  total={avg_lat['total']:.3f}s")
        if grade_result:
            print(f"  accuracy: {grade_result['scores']}")
        print()

        all_results.append({
            "id": sid,
            "text": text,
            "latency_avg": avg_lat,
            "latency_runs": latencies,
            "grade": grade_result,
        })

    if not all_results:
        print("No results collected.")
        return

    n = len(all_results)
    avg = {k: sum(r["latency_avg"][k] for r in all_results) / n
           for k in ("plan", "execute", "summarize", "total")}
    passed = sum(1 for r in all_results if r["grade"] and r["grade"]["passed"])

    print(f"{'─' * 55}")
    print(f"  Summary over {n} scenarios")
    print(f"{'─' * 55}")
    print(f"  Plan:       {avg['plan']:.3f}s  ({avg['plan']/avg['total']*100:.1f}%)")
    print(f"  Execute:    {avg['execute']:.3f}s  ({avg['execute']/avg['total']*100:.1f}%)")
    print(f"  Summarize:  {avg['summarize']:.3f}s  ({avg['summarize']/avg['total']*100:.1f}%)")
    print(f"  Total:      {avg['total']:.3f}s")
    print(f"  Accuracy:   {passed}/{n} passed ({passed/n*100:.1f}%)")
    print(f"{'─' * 55}")

    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps({
        "model": args.model_id,
        "thinking": args.thinking,
        "runs_per_scenario": args.runs,
        "summary": {**avg, "accuracy": f"{passed}/{n}"},
        "scenarios": all_results,
    }, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
