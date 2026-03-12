import logging
import math
from typing import List

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim

from reactxen.agents.evaluation_agent.agent import EvaluationAgent

# from scenario_server.grading.util import ModelCache

# RecentSTmodels = ModelCache()

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


def exact_string_match(
    actual: str, expected: str, case_sensitive: bool = False
) -> bool:
    if not case_sensitive:
        a: str = str(actual).strip().lower()
        e: str = str(expected).strip().lower()
    else:
        a: str = str(actual).strip()
        e: str = str(expected).strip()

    return a == e


def numeric_match(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    try:
        a = float(actual)
        e = float(expected)

        return math.isclose(a, e, rel_tol=tolerance)
    except (ValueError, TypeError) as e:
        logger.exception(f"failed to parse: {e=}")
        return False


def evaluation_agent(
    actual: str, charactistic: str, query: str, trace: str, model_id: int = 16
) -> tuple[bool, list[dict]]:
    try:
        eval_agent = EvaluationAgent(model_id=model_id)

        review = eval_agent.evaluate_response(
            agent_response=actual,
            characteristic_answer=charactistic,
            question=query,
            agent_think=trace,
        )

        overall = False
        if (
            review["task_completion"]
            and review["data_retrieval_accuracy"]
            and review["generalized_result_verification"]
            and review["agent_sequence_correct"]
            and review["clarity_and_justification"]
            and (review["hallucinations"] == False)
        ):
            overall = True

        return overall, [
            {"name": "Task Completion", "value": review["task_completion"]},
            {
                "name": "Data Retrieval Accuracy",
                "value": review["data_retrieval_accuracy"],
            },
            {
                "name": "Generalized Result Verification",
                "value": review["generalized_result_verification"],
            },
            {
                "name": "Agent Sequence Correct",
                "value": review["agent_sequence_correct"],
            },
            {
                "name": "Clarity & Justification",
                "value": review["clarity_and_justification"],
            },
            {
                "name": "Hallucinations",
                "value": review["hallucinations"],
            },
            {
                "name": "Suggestions",
                "value": review.get("suggestions", "No suggestions"),
            },
        ]

    except Exception as e:
        logger.exception(f"exception: {e=}")
        return False, [{"name": "result", "value": False}]


# def cosine_similarity(x:str|List, y:str|List, emedding_model:str="all-MiniLM-L6-v2") -> float | List:
#     """Cosine similarity between strings or lists"""

#     model = RecentSTmodels.get(emedding_model)

#     xembed = model.encode(x)
#     yembed = model.encode(y)

#     simscore = cos_sim(xembed, yembed)

#     # extract the diagonal convert to list
#     score_list = simscore.diagonal().tolist()

#     # return a float or list accoring to the call
#     if len(score_list) == 1:
#         result = float(simscore)
#     else:
#         result = score_list

#     return result
