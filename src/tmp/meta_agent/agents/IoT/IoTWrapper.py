import logging
import json
from iotagent.demo.run_reactreflect import getIoTAgent, IN_CONTEXT, getTools
from meta_agent.utils import save_to_tmp

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


class IoTResponse:
    answer: str
    review: ReviewMessage
    reflection: str
    metric: dict
    trajectory: dict
    final_result: str
    summary: str

def formatJSON(obj):

    if isinstance(obj, IoTResponse):
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
    
    raise ValueError('cannot serialize object to JSON')


class IoTAgentFunctions:

    def request(self, request: str, parent_agent=None, parent_model_id=0):
        # Get the tools and initialize the IoTAgent
        tools, _ = getTools()

        agent = getIoTAgent(
            request,
            # tools=tools,
            # inContext=IN_CONTEXT,
            llm_model_id=parent_model_id,
            reflect_step=1,
            enable_agent_ask=True,
        )

        if parent_agent:
            agent.parent_agent = parent_agent

        # Run the agent's action
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
            review_status = reaction['status']
            review_reasoning = reaction['reasoning']
            review_suggestions = reaction['suggestions']

        # Create and populate IoTResponse object
        retval = IoTResponse()
        retval.answer = answer
        retval.reflection = reflection
        retval.review = ReviewMessage(
            status=review_status,
            reasoning=review_reasoning,
            suggestions=review_suggestions,
        )
        retval.metric = agent.metric
        retval.trajectory = agent.trajectory

        if parent_agent:
            trajectoryJSON = json.dumps(agent.trajectory)
            metricJSON = json.dumps(agent.metric)
            parent_agent.add_step_trajectory_and_metric(trajectoryJSON, metricJSON)
            # metric_file = save_to_tmp(agent.metric, "iotagent_metric_")
            # parent_agent.add_step_trajectory(metric_file)
            # trajectory_file = save_to_tmp(
            #     agent.trajectory, "iotagent_trajectory_"
            # )
            # parent_agent.add_step_metric(trajectory_file)


        # Adding final result and summary to the response
        retval.final_result = answer
        retval.summary = (
            f"I am IoT Agent, and I completed my task. The status field denotes the "
            f"status of execution. I also received feedback from the review agent, "
            f"whose suggestions are included in the review field for further insights."
        )

        # Log the reaction and return the response
        ret = json.dumps(retval, indent=2, default=formatJSON)
        logger.info("IoTAgent Response: %s", ret)

        return ret
