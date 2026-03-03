import logging
import json
from typing import Type, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from meta_agent.agents.RuleLogic.RuleLogicWrapper import (
    RuleLogicAgentFunctions,
    RuleLogicAgentResponse,
)
from meta_agent.utils import save_to_tmp
from reactxen.agents.react.agents import ReactAgent

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# --- Helper function for custom JSON serialization ---
def custom_json(obj):
    if isinstance(obj, RuleLogicAgentResponse):
        return {
            "answer": obj.answer,
            "review": {
                "status": obj.review.get("status", "unknown"),
                "reasoning": obj.review.get("reasoning", "unknown"),
                "suggestions": obj.review.get("suggestions", "unknown"),
            },
            "reflection": obj.reflection,
            "message": (
                f"I am the Rule Logic Agent, and I have completed my task. "
                f"The status of my execution is '{obj.review.get('status', 'unknown')}'. "
                f"I also received feedback from the review agent, whose suggestions "
                f"are included in the review field for further insights."
            ),
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --- Pydantic input schema ---
class RuleLogicAgentInputs(BaseModel):
    request: str = Field(
        description="A Rule Logic Agent request describing the logic of a monitoring rule."
    )


# --- The Tool itself ---
class RuleLogicAgent(BaseTool):
    """Tool to interact with the RuleLogicAgent for rule generation and execution on sensor data."""

    name: str = "RuleLogicAgent"
    description: str = (
        "The RuleAgent can generate executable code from a natural language description "
        "of a monitoring rule. It expects knowledge of available sensors for the asset, "
        "and can execute rules on sensor data stored in a file."
    )
    args_schema: Type[BaseModel] = RuleLogicAgentInputs
    response_format: str = "text"

    rulelogicAgentFunctions: RuleLogicAgentFunctions
    parent_agent: Optional[ReactAgent] = None
    parent_model_id: int = 0

    def _run(self, request: str) -> str:
        logger.info("RuleLogicAgent - Processing request")

        response: RuleLogicAgentResponse = self.rulelogicAgentFunctions.request(
            request=request,
            parent_agent=self.parent_agent,
            parent_model_id=self.parent_model_id,
        )

        serialized_response = json.dumps(response, default=custom_json, indent=2)

        logger.info(f"RuleLogicAgent - Response generated: {serialized_response}")

        return serialized_response
