import logging
import json
from fmsr_agent.agent.react import getFMSRAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

MAX_STEPS = 15


# Defining the structure of ReviewMessage and FMSRResponse
class ReviewMessage:
    status: str
    reasoning: str
    suggestions: str

    def __init__(self, status, reasoning, suggestions):
        self.status = status
        self.reasoning = reasoning
        self.suggestions = suggestions


class FMSRResponse:
    answer: str
    review: ReviewMessage
    reflection: str
    final_result: str
    summary: str
    metric: dict
    trajectory: dict


def formatJSON(obj):

    if isinstance(obj, FMSRResponse):
        return {
            "answer": obj.answer,
            "review": {
                "status": obj.review.status,
                "reasoning": obj.review.reasoning,
                "suggestions": obj.review.suggestions,
            },
            "final_result": obj.final_result,
            "summary": obj.summary,
        }

    raise ValueError("cannot serialize object to JSON")


class FMSRAgentFunctions:

    def request(self, request: str, parent_agent=None, parent_model_id=0):
        # Get the react reflect agent based on the request and parent_model_id

        fmsrAgent = getFMSRAgent(
            request,
            llm_model_id=parent_model_id,
            reflect_step=1,
            enable_agent_ask=True,
        )

        if parent_agent:
            fmsrAgent.parent_agent = parent_agent

        # Run the agent's action
        reaction = fmsrAgent.run()

        # Prepare the answer, reflection, and review fields
        answer = "Agent failed to generate final answer"
        if fmsrAgent.answer.strip() != "":
            answer = fmsrAgent.answer

        # Reflection field (get the most recent reflection)
        reflection = "None"
        if len(fmsrAgent.reflections) > 0 and fmsrAgent.reflections[-1].strip() != "":
            reflection = fmsrAgent.reflections[-1]

        # Review field (get the most recent review, if available)
        review_status = "None"
        review_reasoning = "None"
        review_suggestions = "None"
        if reaction:
            review_status = reaction["status"]
            review_reasoning = reaction["reasoning"]
            review_suggestions = reaction["suggestions"]

        # Create and populate FMSRResponse object
        retval = FMSRResponse()
        retval.answer = answer
        retval.reflection = reflection
        retval.review = ReviewMessage(
            status=review_status,
            reasoning=review_reasoning,
            suggestions=review_suggestions,
        )
        retval.metric = fmsrAgent.metric
        retval.trajectory = fmsrAgent.trajectory

        # Adding final result and summary to the response
        retval.final_result = answer
        retval.summary = (
            f"I am FMSR Agent, and I completed my task. The status field denotes the "
            f"status of execution. I also received feedback from the review agent, "
            f"whose suggestions are included in the review field for further insights."
        )

        # Log the reaction and return the response
        ret = json.dumps(retval, indent=2, default=formatJSON)
        logger.info("FMSRAgent Response: %s", ret)

        return ret
