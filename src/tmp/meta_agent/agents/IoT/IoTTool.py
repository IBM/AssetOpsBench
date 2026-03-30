from typing import Type, Optional
from meta_agent.agents.IoT.IoTWrapper import IoTAgentFunctions, IoTResponse
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from reactxen.agents.react.agents import ReactAgent
from meta_agent.utils import save_to_tmp
import logging
import traceback
import json

logger = logging.getLogger(__name__)


def custom_json(obj):
    if isinstance(obj, IoTResponse):
        return {
            "answer": obj.answer,
            # 'metric': save_to_tmp(obj.metric, "iotagent_metric_"),
            # 'trajectory': save_to_tmp(obj.trajectory, "iotagent_trajectory_"),
            "review": {
                "status": obj.review["status"],
                "reasoning": obj.review["reasoning"],
                "suggestions": obj.review["suggestions"],
            },
            "reflection": obj.reflection,
            "message": (
                f"I am IoT Agent, and I have completed my task. "
                f"The status of my execution is '{obj.review['status']}'. "
                f"I also received a review from the reflection agent; "
                f"suggestions are included in the review field for further insights."
            ),
        }
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


class IoTAgentInputs(BaseModel):
    request: str = Field(
        description="An Internet of Things (IoT) request for information."
    )


class IoTAgent(BaseTool):
    """Tool to make requests to the IoTAgent."""

    name: str = "IoTAgent"
    description: str = (
        "The IoTAgent provides information about IoT sites, asset details, sensor data, "
        "and retrieves historical data and metadata for various assets and equipment."
    )
    args_schema: Type[BaseModel] = IoTAgentInputs
    response_format: str = "text"

    iotAgentFunctions: IoTAgentFunctions
    parent_agent: Optional[ReactAgent] = None
    parent_model_id: Optional[int] = 0

    def _run(self, request: str) -> str:
        logger.info("=== IoTAgent: Handling Request ===")
        logger.debug(f"Request: {request}")

        try:
            response: IoTResponse = self.iotAgentFunctions.request(
                request=request,
                parent_agent=self.parent_agent,
                parent_model_id=self.parent_model_id,
            )
            logger.info("=== IoTAgent: Successfully received response ===")
            logger.debug(
                "Response (raw): %s",
                json.dumps(response, default=custom_json, indent=2),
            )

            return json.dumps(response, default=custom_json)

        except Exception as e:
            logger.error(f"=== IoTAgent: Error occurred ===\n{e}")
            logger.error(traceback.format_exc())

            # Return a failure status instead of raising an exception
            failure_response = {
                "status": "failure",
                "error": str(e),
                "message": "An error occurred while processing the request.",
            }
            return json.dumps(failure_response, indent=2)
