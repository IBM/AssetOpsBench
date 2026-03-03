from typing import Type, Optional
from meta_agent.agents.WorkOrder.WorkOrderWrapper import WOAgentFunctions, WOResponse
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from reactxen.agents.react.agents import ReactAgent
from meta_agent.utils import save_to_tmp
import logging
import traceback
import json

logger = logging.getLogger(__name__)


def custom_json(obj):
    if isinstance(obj, WOResponse):
        return {
            "answer": obj.answer,
            # 'metric': save_to_tmp(obj.metric, "woagent_metric_"),
            # 'trajectory': save_to_tmp(obj.trajectory, "woagent_trajectory_"),
            "review": {
                "status": obj.review["status"],
                "reasoning": obj.review["reasoning"],
                "suggestions": obj.review["suggestions"],
            },
            "reflection": obj.reflection,
            "message": (
                f"I am Work Order Agent, and I have completed my task. "
                f"The status of my execution is '{obj.review['status']}'. "
                f"I also received a review from the reflection agent; "
                f"suggestions are included in the review field for further insights."
            ),
        }
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


class WOAgentInputs(BaseModel):
    request: str = Field(
        description="A Work Order (WO) request for asset-related information."
    )


class WOAgent(BaseTool):
    """Tool to make requests to the WOAgent."""

    name: str = "WOAgent"
    description: str = (
        "The Work Order (WO) agent can retrieve, analyze, and generate work orders for equipment "
        "based on historical data, anomalies, alerts, and performance metrics, offering recommendations "
        "for preventive and corrective actions, including bundling, prioritization, and predictive maintenance."
    )
    args_schema: Type[BaseModel] = WOAgentInputs
    response_format: str = "text"

    tsfmAgentFunctions: WOAgentFunctions
    parent_agent: Optional[ReactAgent] = None
    parent_model_id: Optional[int] = 0

    def _run(self, request: str) -> str:
        logger.info("=== WOAgent: Handling Request ===")
        logger.debug(f"Request: {request}")

        try:
            response: WOResponse = self.tsfmAgentFunctions.request(
                request=request,
                parent_agent=self.parent_agent,
                parent_model_id=self.parent_model_id,
            )
            logger.info("=== WOAgent: Successfully received response ===")
            logger.debug(
                "Response (raw): %s",
                json.dumps(response, default=custom_json, indent=2),
            )

            return json.dumps(response, default=custom_json)

        except Exception as e:
            logger.error(f"=== WOAgent: Error occurred ===\n{e}")
            logger.error(traceback.format_exc())

            failure_response = {
                "status": "failure",
                "error": str(e),
                "message": "An error occurred while processing the request.",
            }
            return json.dumps(failure_response, indent=2)
