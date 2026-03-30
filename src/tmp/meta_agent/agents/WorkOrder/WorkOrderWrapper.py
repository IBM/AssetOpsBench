import logging
import json

from woagent.demo.run_agent_with_react_review_reflect_kdd import getWOAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

MAX_STEPS = 15


class ReviewMessage:
    status: str
    reasoning: str
    suggestions: str

    def __init__(self, status, reasoning, suggestions):
        self.status = status
        self.reasoning = reasoning
        self.suggestions = suggestions


class WOResponse:
    answer: str
    review: ReviewMessage
    reflection: str
    metric: dict
    trajectory: dict
    final_result: str
    summary: str


def formatJSON(obj):

    if isinstance(obj, WOResponse):
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


class WOAgentFunctions:

    def request(self, request: str, parent_agent=None, parent_model_id=0):
        # Initialize the WOAgent using the provided query

        woAgent = getWOAgent(
            request,
            llm_model_id=parent_model_id,
            reflect_step=1,
            enable_agent_ask=True,
        )

        if parent_agent:
            woAgent.parent_agent = parent_agent

        # Run the agent's action
        reaction = woAgent.run()

        # Prepare the answer and reflection fields
        answer = "Agent failed to generate final answer"
        if woAgent.answer.strip():
            answer = woAgent.answer

        reflection = "None"
        if woAgent.reflections and woAgent.reflections[-1].strip():
            reflection = woAgent.reflections[-1]

        review_status = "None"
        review_reasoning = "None"
        review_suggestions = "None"
        if reaction:
            review_status = reaction['status']
            review_reasoning = reaction['reasoning']
            review_suggestions = reaction['suggestions']

        # Create and populate WOResponse object
        retval = WOResponse()
        retval.answer = answer
        retval.reflection = reflection
        retval.review = ReviewMessage(
            status=review_status,
            reasoning=review_reasoning,
            suggestions=review_suggestions,
        )
        retval.metric = woAgent.metric
        retval.trajectory = woAgent.trajectory

        # Adding final result and summary to the response
        retval.final_result = answer
        retval.summary = (
            f"I am Work Order Agent, and I completed my task. The status field denotes the "
            f"status of execution. I also received feedback from the review agent, "
            f"whose suggestions are included in the review field for further insights."
        )

        # Log the response and return
        ret = json.dumps(retval, indent=2, default=formatJSON)
        logger.info("WOAgent Response: %s", ret)
        return ret
