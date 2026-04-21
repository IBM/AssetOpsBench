from abc import ABC, abstractmethod
from typing import List
from pydantic import Field
from agent_hive.task import Task


class Workflow(ABC):
    """Base class for all workflows."""

    tasks: List[Task] = Field(description="List of tasks to execute.")
    memory: List[str] = Field(default=[], description="Memory of the workflow.")
    max_memory: int = Field(default=10, description="Maximum memory size of the workflow.")

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
