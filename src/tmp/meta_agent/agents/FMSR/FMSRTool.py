from typing import Type, Optional
from meta_agent.agents.FMSR.FMSRWrapper import FMSRAgentFunctions, FMSRResponse
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from meta_agent.utils import save_to_tmp
from reactxen.agents.react.agents import ReactAgent
import logging
import traceback
import json

logger = logging.getLogger(__name__)


def custom_json(obj):
    if isinstance(obj, FMSRResponse):
        return {
            "answer": obj.answer,
            # 'metric': save_to_tmp(obj.metric, "fmsragent_metric_"),
            # 'trajectory': save_to_tmp(obj.trajectory, "fmsragent_trajectory_"),
            "review": {
                "status": obj.review["status"],
                "reasoning": obj.review["reasoning"],
                "suggestions": obj.review["suggestions"],
            },
            "reflection": obj.reflection,
            "message": (
                f"I am FMSR Agent, and I have completed my task. "
                f"The status of my execution is '{obj.review['status']}'. "
                f"I also received a review from the reflection agent; "
                f"suggestions are included in the review field for further insights."
            ),
        }
    raise TypeError(f"Cannot serialize object of type {type(obj)}")

class FMSRAgentInputs(BaseModel):
    request: str = Field(
        description="A Failure Mode Sensor Relevance (FMSR) request for information."
    )

class FMSRAgent(BaseTool):
    """Tool to make requests to the FMSRAgent."""

    name: str = "FMSRAgent"
    description: str = (
        "The FMSRAgent provides information about failure modes, the mapping between failure modes and sensors, "
        "detection strategies for assets like chillers and wind turbines, and can generate detection recipes, "
        "anomaly detection plans, and machine learning recipes for specific failures."
    )
    args_schema: Type[BaseModel] = FMSRAgentInputs
    response_format: str = "text"

    fmsrAgentFunctions: FMSRAgentFunctions
    parent_agent: Optional[ReactAgent] = None
    parent_model_id: Optional[int] = 0

    def _run(self, request: str) -> str:
        logger.info("=== FMSRAgent: Handling Request ===")
        logger.debug(f"Request: {request}")

        try:
            response: FMSRResponse = self.fmsrAgentFunctions.request(
                request=request,
                parent_agent=self.parent_agent,
                parent_model_id=self.parent_model_id,
            )
            logger.info("=== FMSRAgent: Successfully received response ===")
            logger.debug("Response (raw): %s", json.dumps(response, default=custom_json, indent=2))

            return json.dumps(response, default=custom_json)

        except Exception as e:
            logger.error(f"=== FMSRAgent: Error occurred ===\n{e}")
            logger.error(traceback.format_exc())

            # Return a failure status instead of raising an exception
            failure_response = {
                "status": "failure",
                "error": str(e),
                "message": "An error occurred while processing the request."
            }
            return json.dumps(failure_response, indent=2)
        
