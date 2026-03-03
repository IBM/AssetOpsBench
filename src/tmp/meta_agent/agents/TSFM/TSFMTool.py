from typing import Type
from meta_agent.agents.TSFM.TSFMWrapper import (
    TSFMAgentFunctions,
    ReviewMessage,
    TSFMResponse,
)
from pydantic import BaseModel, Field
import logging
from typing import Type
from langchain_core.tools import BaseTool
import json
from meta_agent.utils import save_to_tmp
import traceback

logger: logging.Logger = logging.getLogger(__name__)
from reactxen.agents.react.agents import ReactAgent

def custom_json(obj):

    if isinstance(obj, TSFMResponse):

        return {
            "answer": obj.answer,
            # 'metric': save_to_tmp(obj.metric, "tsfmagent_metric_"),
            # 'trajectory': save_to_tmp(obj.trajectory, "tsfmagent_trajectory_"),
            "review": {
                "status": obj.review["status"],
                "reasoning": obj.review["reasoning"],
                "suggestions": obj.review["suggestions"],
            },
            "reflection": obj.reflection,
            "message": (
                f"I am TSFM Agent, and I have completed my task. "
                f"The status of my execution is '{obj.review['status']}'. "
                f"I also received a review(reflect) from the review(reflect) agent, whose suggestions "
                f"are included in the review field for further insights."
            ),
        }

    raise TypeError(f"Cannot serialize object of {type(obj)}")


class TSFMAgentInputs(BaseModel):
    request: str = Field(
        description="a Time Series Foundation Model request for information"
    )


class TSFMAgent(BaseTool):
    """Tool to make requests to TSFMAgent"""

    name: str = "TSFMAgent"
    description: str = (
        "The Time Series Agent can assist with time series analysis, forecasting, anomaly detection, and model selection, and supports pretrained models, context length specifications, and regression tasks for various time series data."
    )
    args_schema: Type[BaseModel] = TSFMAgentInputs
    response_format: str = "text"
    tsfmAgentFunctions: TSFMAgentFunctions
    parent_agent: ReactAgent = None
    parent_model_id: int = 0

    def _run(self, request: str) -> str:
        logger.info("=== TSFMAgent: Handling Request ===")
        logger.debug(f"Request: {request}")

        try:
            response: TSFMResponse = self.tsfmAgentFunctions.request(
                request=request,
                parent_agent=self.parent_agent,
                parent_model_id=self.parent_model_id,
            )

            logger.info("=== TSFMAgent: Successfully received response ===")
            logger.debug("Response (raw): %s", json.dumps(response, default=custom_json, indent=2))

            return json.dumps(response, default=custom_json)
        except Exception as e:
            logger.error(f"=== TSFMAgent: Error occurred ===\n{e}")
            logger.error(traceback.format_exc())

            # Return a failure status instead of raising an exception
            failure_response = {
                "status": "failure",
                "error": str(e),
                "message": "An error occurred while processing the request."
            }
            return json.dumps(failure_response, indent=2)


