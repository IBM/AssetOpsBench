from abc import ABC, abstractmethod
from typing import Any, List

from langchain.tools import BaseTool
from pydantic import Field


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    name: str = Field(description="Name of the agent.")
    description: str = Field(description="Description of the agent.")
    llm: str = Field(description="LLM used by the agent.")
    memory: List[str] = Field(default_factory=list, description="Memory of the agent.")
    tools: List[BaseTool] = Field(
        default_factory=list, description="Tools the agent is limited to use."
    )

    @abstractmethod
    def execute_task(self, *args, **kwargs):
        pass

    def __str__(self):
        tool_names = (
            ", ".join(str(tool.name) for tool in self.tools) if self.tools else "None"
        )
        return f"{self.__class__.__name__}(name={self.name}, description={self.description}, llm={self.llm}, tools=[{tool_names}])"
