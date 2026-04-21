def compute_leaderboard(results: list[dict]) -> dict:
    summary = {}

    for result in results:
        model = result["model"]

        if model not in summary:
            summary[model] = {
                "total_questions": 0,
                "successful_runs": 0,
                "total_score": 0.0,
                "total_latency": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "live_runs": 0,
            }

        summary[model]["total_questions"] += 1
        summary[model]["total_latency"] += result["latency"]
        summary[model]["total_score"] += result["score"]
        summary[model]["total_input_tokens"] += result["estimated_input_tokens"]
        summary[model]["total_output_tokens"] += result["estimated_output_tokens"]
        summary[model]["total_tokens"] += result["estimated_total_tokens"]

        if result["success"]:
            summary[model]["successful_runs"] += 1

        if result["mode"] == "live":
            summary[model]["live_runs"] += 1

    for model, stats in summary.items():
        total = stats["total_questions"]
        stats["success_rate"] = stats["successful_runs"] / total if total else 0.0
        stats["accuracy"] = stats["total_score"] / total if total else 0.0
        stats["average_latency"] = stats["total_latency"] / total if total else 0.0
        stats["average_input_tokens"] = (
            stats["total_input_tokens"] / total if total else 0.0
        )
        stats["average_output_tokens"] = (
            stats["total_output_tokens"] / total if total else 0.0
        )
        stats["average_total_tokens"] = stats["total_tokens"] / total if total else 0.0

    return summary


def print_question_results(results: list[dict]) -> None:
    print("\nPer-question Results")
    print("-" * 210)
    print(
        f"{'Question':<9}"
        f"{'Model':<62}"
        f"{'Mode':<12}"
        f"{'Success':<10}"
        f"{'Score':<8}"
        f"{'Latency':<10}"
        f"{'In Tok':<10}"
        f"{'Out Tok':<10}"
        f"{'Total Tok':<12}"
        f"{'Evaluation':<58}"
        f"{'Answer Preview'}"
    )
    print("-" * 210)

    for result in results:
        preview = result["answer"].replace("\n", " ").strip()
        if len(preview) > 35:
            preview = preview[:32] + "..."

        evaluation = result.get("evaluation_summary", "")
        if len(evaluation) > 55:
            evaluation = evaluation[:52] + "..."

        print(
            f"{result['question_id']:<9}"
            f"{result['model']:<62}"
            f"{result['mode']:<12}"
            f"{str(result['success']):<10}"
            f"{result['score']:<8.2f}"
            f"{result['latency']:<10.2f}"
            f"{result['estimated_input_tokens']:<10}"
            f"{result['estimated_output_tokens']:<10}"
            f"{result['estimated_total_tokens']:<12}"
            f"{evaluation:<58}"
            f"{preview}"
        )


def print_leaderboard(summary: dict) -> None:
    print("\nLeaderboard Summary")
    print("-" * 137)
    print(
        f"{'Model':<62}"
        f"{'Questions':<12}"
        f"{'Average Score':<16}"
        f"{'Average Latency':<18}"
        f"{'Average Tokens':<17}"
        f"{'Score/1k Tok':<17}"
        f"{'Live':<8}"
    )
    print("-" * 137)

    for model, stats in summary.items():
        average_tokens = stats["average_total_tokens"]
        quality_weighted_score_per_1k_tokens = (
            (stats["accuracy"] * stats["accuracy"]) / (average_tokens / 1000)
            if average_tokens
            else 0.0
        )
        print(
            f"{model:<62}"
            f"{stats['total_questions']:<12}"
            f"{stats['accuracy']:<16.2%}"
            f"{stats['average_latency']:<18.2f}"
            f"{average_tokens:<17.1f}"
            f"{quality_weighted_score_per_1k_tokens:<17.2f}"
            f"{stats['live_runs']:<8}"
        )
