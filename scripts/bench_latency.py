#!/usr/bin/env python3
"""Benchmark latency and accuracy for multi-agent scenarios.

Each scenario is executed once: latency recorded + accuracy graded via LLM-as-judge.

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "openai/Qwen/Qwen2.5-7B-Instruct"  # requires LITELLM_BASE_URL pointing to local vLLM

# ── LLM-as-judge prompt ───────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating an AI agent's response to an industrial asset operations query.

Question: {question}
Expected outcome: {characteristic_form}
Agent execution trace: {trace}
Agent response: {answer}

Evaluate the response on the following six dimensions. Use the expected outcome as the ground truth.

1. task_completion: Did the agent complete the requested task as described in the expected outcome? If the expected outcome says data exists and a result should be produced, but the agent says data is missing or unavailable, this is False.

2. data_retrieval_accuracy: Was the correct data retrieved with the correct parameters (asset, location, time range, sensor)? If the agent failed to retrieve data that the expected outcome confirms should exist, this is False.

3. generalized_result_verification: Does the agent's result match what the expected outcome says should have happened? If the expected outcome states that anomalies were detected, a forecast was produced, or specific data was found, but the agent reports failure or missing data, this is False.

4. agent_sequence_correct: Were all required steps executed in the correct order as implied by the expected outcome? Use the execution trace to verify the actual sequence of tool calls. If the expected outcome specifies multiple steps and the trace shows the agent skipped or failed any of them, this is False.

5. clarity_and_justification: Is the response clearly written and does it justify its conclusions?

6. hallucinations: Did the agent state facts that contradict the expected outcome or are not supported by the data? If the expected outcome confirms that data exists (e.g., 400+ records, anomalies detected) but the agent claims there is no data or the asset does not exist, this is True (hallucination occurred).

Respond with a JSON object only, no explanation:
{{"task_completion": true, "data_retrieval_accuracy": true, "generalized_result_verification": true, "agent_sequence_correct": true, "clarity_and_justification": true, "hallucinations": false}}
"""


# ── dataset loading ───────────────────────────────────────────────────────────

HF_DATASET = "ibm-research/AssetOpsBench"

def load_scenarios_hf(split: str = "train") -> list[dict]:
    """Load all scenarios from HuggingFace ibm-research/AssetOpsBench.

    Expected dataset columns: id, text, characteristic_form.
    """
    from datasets import load_dataset  # type: ignore

    print(f"Loading scenarios from HuggingFace: {HF_DATASET} (split={split}) ...")
    ds = load_dataset(HF_DATASET, split=split)
    scenarios = [dict(row) for row in ds]
    print(f"Loaded {len(scenarios)} scenarios from {HF_DATASET}\n")
    return scenarios


def load_scenarios_local() -> list[dict]:
    """Load all scenarios from the local chiller_utterance.json file (fallback)."""
    data_file = REPO_ROOT / "src" / "scenarios" / "local" / "chiller_utterance.json"
    with open(data_file) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios from {data_file.name}\n")
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


def grade_custom(question: str, characteristic_form: str, answer: str, trace: str = "", max_retries: int = 3) -> dict:
    """LLM-as-judge using custom prompt (default mode)."""
    llm = _get_judge_llm()
    prompt = _JUDGE_PROMPT.format(
        question=question,
        characteristic_form=characteristic_form,
        answer=answer,
        trace=trace or "(not available)",
    )

    for attempt in range(1, max_retries + 1):
        try:
            raw = llm.generate(prompt)
            text = raw.strip()
            if "```" in text:
                lines = text.splitlines()
                text = "\n".join(l for l in lines if not l.strip().startswith("```"))
            start, end = text.find("{"), text.rfind("}") + 1
            scores = json.loads(text[start:end]) if start != -1 else {}
            if scores:
                return {"scores": scores}
            raise ValueError("empty scores")
        except Exception as e:
            print(f" [judge attempt {attempt}/{max_retries} failed: {e}]", end="", flush=True)
            if attempt == max_retries:
                return {"scores": {}}

    return {"scores": {}}


def grade_reactxen(question: str, characteristic_form: str, answer: str, trace: str = "") -> dict:
    """Grade using the original EvaluationAgent from reactxen (scenario-server)."""
    sys.path.insert(0, str(REPO_ROOT / "aobench" / "scenario-server" / "src"))
    try:
        from reactxen.agents.evaluation_agent.agent import EvaluationAgent
    except ImportError:
        print("[ERROR] reactxen not installed. Run: pip install git+https://github.com/IBM/ReActXen.git", file=sys.stderr)
        sys.exit(1)

    eval_agent = EvaluationAgent(model_id=16)
    review = eval_agent.evaluate_response(
        agent_response=answer,
        characteristic_answer=characteristic_form,
        question=question,
        agent_think=trace or "",
    )
    return {"scores": review}


def grade(question: str, characteristic_form: str, answer: str, trace: str = "", mode: str = "custom") -> dict:
    """Call grader based on mode ('custom' or 'reactxen')."""
    if mode == "reactxen":
        return grade_reactxen(question, characteristic_form, answer, trace)
    return grade_custom(question, characteristic_form, answer, trace)


# ── per-scenario worker ───────────────────────────────────────────────────────

_print_lock = threading.Lock()


def process_scenario(idx: int, total: int, scenario: dict, args: argparse.Namespace) -> dict | None:
    """Run one scenario once: record latency and grade the answer.

    Output is buffered and printed atomically under _print_lock.
    """
    sid = scenario["id"]
    text = scenario["text"]
    characteristic_form = scenario.get("characteristic_form", "")

    lines: list[str] = []
    lines.append(f"[{idx}/{total}] id={sid}: {text}")
    lines.append(f"  expected: {characteristic_form}")

    grade_result: dict | None = None
    latency: dict | None = None

    output = run_scenario(text, args.model_id, thinking=args.thinking)
    if output is None:
        lines.append("  [FAILED]")
    else:
        latency = output.get("latency") or None
        if latency:
            lines.append(f"  latency: plan={latency['plan']:.2f}s  execute={latency['execute']:.2f}s  "
                         f"summarize={latency['summarize']:.2f}s  total={latency['total']:.2f}s")
        if output.get("answer"):
            lines.append(f"  answer: {output['answer']}")
            lines.append("  grading...")
            trace = json.dumps({
                "plan": output.get("plan", []),
                "history": output.get("history", []),
            }, indent=2)
            grade_result = grade(text, characteristic_form, output["answer"], trace, mode=args.grade_mode)
            lines.append(f"  graded: {grade_result['scores']}")

    with _print_lock:
        print("\n".join(lines) + "\n", flush=True)

    if latency is None:
        return None

    return {
        "id": sid,
        "text": text,
        "latency": latency,
        "grade": grade_result,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark plan-execute on multi-agent scenarios 501-520.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help=f"LiteLLM model string (default: {DEFAULT_MODEL})")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode in the planning phase.")
    parser.add_argument("--grade-mode", default="custom", choices=["custom", "reactxen"], help="Grading mode: 'custom' (LLM-as-judge prompt) or 'reactxen' (original EvaluationAgent).")
    parser.add_argument("--local", action="store_true", help="Load scenarios from local file instead of HuggingFace.")
    parser.add_argument("--split", default="train", help="HuggingFace dataset split to use (default: train).")
    parser.add_argument("--id-min", type=int, default=400, help="Minimum scenario id, inclusive (default: 400).")
    parser.add_argument("--id-max", type=int, default=446, help="Maximum scenario id, inclusive (default: 446).")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of scenarios to run in parallel (default: 1).")
    args = parser.parse_args()

    if args.local:
        scenarios = load_scenarios_local()
    else:
        scenarios = load_scenarios_hf(split=args.split)

    scenarios = [s for s in scenarios if args.id_min <= int(s["id"]) <= args.id_max]

    print(f"Model:      {args.model_id}")
    print(f"Thinking:   {'enabled' if args.thinking else 'disabled'}")
    print(f"ID range:   {args.id_min}–{args.id_max}  ({len(scenarios)} scenarios)")
    print(f"Batch size: {args.batch_size}")
    print(f"Grade mode: {args.grade_mode}\n")

    total = len(scenarios)
    all_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
        futures = {
            pool.submit(process_scenario, i, total, scenario, args): i
            for i, scenario in enumerate(scenarios, 1)
        }
        # collect in completion order; final summary is sorted by id
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("No results collected.")
        return

    all_results.sort(key=lambda r: r["id"])
    n = len(all_results)
    avg = {k: sum(r["latency"][k] for r in all_results) / n
           for k in ("plan", "execute", "summarize", "total")}

    dims = ["task_completion", "data_retrieval_accuracy", "generalized_result_verification",
            "agent_sequence_correct", "clarity_and_justification", "hallucinations"]
    graded = [r for r in all_results if r["grade"]]
    ng = len(graded)
    dim_scores = {
        d: sum(1 for r in graded if r["grade"]["scores"].get(d, False)) / ng if ng else 0
        for d in dims
    }

    print(f"{'─' * 55}")
    print(f"  Summary over {n} scenarios ({ng} graded)")
    print(f"{'─' * 55}")
    print(f"  Plan:       {avg['plan']:.3f}s  ({avg['plan']/avg['total']*100:.1f}%)")
    print(f"  Execute:    {avg['execute']:.3f}s  ({avg['execute']/avg['total']*100:.1f}%)")
    print(f"  Summarize:  {avg['summarize']:.3f}s  ({avg['summarize']/avg['total']*100:.1f}%)")
    print(f"  Total:      {avg['total']:.3f}s")
    print(f"  ── Accuracy per dimension ──")
    for d, score in dim_scores.items():
        # hallucinations: lower is better
        label = f"{score*100:.0f}% hallucinated" if d == "hallucinations" else f"{score*100:.0f}% passed"
        print(f"    {d}: {label}")
    print(f"{'─' * 55}")

    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps({
        "model": args.model_id,
        "thinking": args.thinking,
        "summary": {**avg, "dim_scores": dim_scores},
        "scenarios": all_results,
    }, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
