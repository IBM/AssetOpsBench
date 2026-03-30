import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import json

from tsfmagent.agents.tsfmagent.tsfm_agent import getTSFMAgent

# from tsfm_agent.tools.tsfm.tool import TSFMForecastingRun
# from tsfm_agent.tools.tsfm.TSFMWrapper import TSFMForecastWrapper
# from tsfm_agent.tools.tsad.tool import TimeSeriesAnomalyDetectionRun
# from tsfm_agent.tools.tsad.TimeSeriesAnomalyDetectionWrapper import TimeSeriesAnomalyDetectionConformalWrapper

logger: logging.Logger = logging.getLogger(__name__)

MAX_STEPS = 15


class ReviewMessage:
    status: str
    reasoning: str
    suggestions: str

    def __init__(self, status, reasoning, suggestions):
        self.status = status
        self.reasoning = reasoning
        self.suggestions = suggestions


class TSFMResponse:
    answer: str
    review: ReviewMessage
    reflection: str
    metric: dict
    trajectory: dict
    final_result: str
    summary: str


def formatJSON(obj):

    if isinstance(obj, TSFMResponse):
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


class TSFMAgentFunctions:

    def request(self, request: str, parent_agent=None, parent_model_id=0):

        agent = getTSFMAgent(
            request, llm_model_id=parent_model_id, reflect_step=1, enable_agent_ask=True
        )  # , react_step=2, reflect_step=2)
        if parent_agent:
            agent.parent_agent = parent_agent
        reaction = agent.run()

        # Prepare the answer, reflection, and review fields
        answer = "Agent failed to generate final answer"
        if agent.answer.strip():
            answer = agent.answer

        reflection = "None"
        if len(agent.reflections) > 0 and agent.reflections[-1].strip():
            reflection = agent.reflections[-1]

        review_status = "None"
        review_reasoning = "None"
        review_suggestions = "None"
        if reaction:
            review_status = reaction["status"]
            review_reasoning = reaction["reasoning"]
            review_suggestions = reaction["suggestions"]

        # Create and populate IoTResponse object
        retval = TSFMResponse()
        retval.answer = answer
        retval.reflection = reflection
        retval.review = ReviewMessage(
            status=review_status,
            reasoning=review_reasoning,
            suggestions=review_suggestions,
        )
        retval.metric = agent.metric
        retval.trajectory = agent.trajectory

        # Adding final result and summary to the response
        retval.final_result = answer
        retval.summary = (
            f"I am TSFM Agent, and I completed my task. The status field denotes the "
            f"status of execution. I also received feedback from the review agent, "
            f"whose suggestions are included in the review field for further insights."
        )

        # Log the reaction and return the response
        ret = json.dumps(retval, indent=2, default=formatJSON)
        logger.info("TSFMAgent Response: %s", ret)

        return ret
