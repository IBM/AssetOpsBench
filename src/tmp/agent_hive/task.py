from typing import List, Optional

from pydantic import Field

from agent_hive.agents.base_agent import BaseAgent


class Task:
    description: str = Field(description="Description of the actual task.")
    agents: List[BaseAgent] = Field(description="Agents responsible for execution the task.")
    expected_output: Optional[str] = Field(default=None,
                                           description="Clear definition of expected output for the task.")
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )

    def __init__(self, description: str, agents: List[BaseAgent], expected_output: Optional[str] = None,
                 context: Optional[List['Task']] = None):
        self.description = description
        self.agents = agents
        self.expected_output = expected_output
        self.context = context

    def __str__(self):
        return f"Task(description={self.description}, agents={self.agents}, expected_output={self.expected_output}, context={self.context})"
