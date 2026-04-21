import subprocess
import time

from evaluation.evaluator import load_questions, evaluate_answer_details
from evaluation.leaderboard import (
    compute_leaderboard,
    print_leaderboard,
    print_question_results,
)

MODEL_IDS = [
    "watsonx/ibm/granite-4-h-small",
    "watsonx/mistral-large-2512",
    "watsonx/mistralai/mistral-medium-2505",
    "watsonx/mistralai/mistral-small-3-1-24b-instruct-2503",
    "watsonx/openai/gpt-oss-120b",
]


def extract_final_answer(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def estimate_tokens(text: str) -> int:
    # Simple estimate for leaderboard comparison when provider token usage is unavailable.
    return max(1, round(len(text.split()) * 1.3)) if text.strip() else 0


def add_token_estimates(result: dict, question: str) -> None:
    result["estimated_input_tokens"] = estimate_tokens(question)
    result["estimated_output_tokens"] = estimate_tokens(result["answer"])
    result["estimated_total_tokens"] = (
        result["estimated_input_tokens"] + result["estimated_output_tokens"]
    )


def run_plan_execute_live(question: str, model_id: str) -> dict:
    start_time = time.time()

    try:
        completed = subprocess.run(
            ["uv", "run", "plan-execute", "--model-id", model_id, question],
            capture_output=True,
            text=True,
            check=True,
        )

        latency = time.time() - start_time
        raw_output = completed.stdout.strip()
        answer = extract_final_answer(raw_output)

        return {
            "answer": answer,
            "success": True,
            "latency": latency,
            "mode": "live",
        }

    except subprocess.CalledProcessError as error:
        latency = time.time() - start_time
        raw_output = (error.stdout or "").strip()
        answer = extract_final_answer(raw_output)

        return {
            "answer": answer,
            "success": False,
            "latency": latency,
            "mode": "live",
        }


def main() -> None:
    questions = load_questions()
    results = []

    for question in questions:
        question_id = question["id"]
        question_text = question["question"]

        for model_id in MODEL_IDS:
            print(
                f"\nRunning live plan_execute for {question_id} "
                f"with {model_id}: {question_text}"
            )
            model_result = run_plan_execute_live(question_text, model_id)
            evaluation = evaluate_answer_details(model_result["answer"], question)
            model_result["score"] = evaluation["score"]
            model_result["evaluation_summary"] = evaluation["summary"]
            model_result["question_id"] = question_id
            model_result["model"] = model_id
            add_token_estimates(model_result, question_text)
            results.append(model_result)

    print_question_results(results)
    summary = compute_leaderboard(results)
    print_leaderboard(summary)

    print(
        "\nNote: all rows use the live plan_execute workflow. Token counts are "
        "estimated because the current CLI output does not expose provider usage."
    )


if __name__ == "__main__":
    main()
