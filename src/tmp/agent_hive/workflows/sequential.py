import json
from typing import List

from pydantic import Field

from agent_hive.enum import ContextType
from agent_hive.task import Task
from agent_hive.workflows.base_workflow import Workflow
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class SequentialWorkflow(Workflow):
    """
    This class represents a sequential agentic workflow, where each task is executed in order. And each task is
    assigned to a specific agent.

    Example:
        agent1 = ...
        agent2 = ...

        task1 = Task(..., agents=[agent1], ...)
        task2 = Task(..., agents=[agent2], ...)

        workflow = SequentialWorkflow(tasks=[task1, task2], ...)
        workflow.run()

    """

    context_type: ContextType = Field(
        default=ContextType.DISABLED, description="Type of context to use."
    )

    def __init__(
        self, tasks: List[Task], context_type: ContextType = ContextType.DISABLED
    ):
        self.tasks = tasks
        self.context_type = context_type
        self.memory = []
        self.max_memory = 10
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        for i, task in enumerate(self.tasks):
            if task.agents is None or len(task.agents) == 0:
                raise ValueError("Task must have at least one agent")
            if len(task.agents) > 1:
                raise NotImplementedError(
                    "SequentialWorkflow only supports one agent per task"
                )
            if self.context_type == ContextType.SELECTED:
                if isinstance(task.context, list):
                    for context_task in task.context:
                        if context_task not in self.tasks[:i]:
                            raise ValueError(
                                "task.context must be a list of Task objects that are part of the workflow"
                            )

    def run(self):
        self.memory = []
        for i, task in enumerate(self.tasks):
            task_no = i + 1
            logger.info(f"Task {task_no}: {task.description}")
            assigned_agent = task.agents[0]

            if self.context_type == ContextType.DISABLED:
                user_input = task.description
                response = assigned_agent.execute_task(user_input)

            elif self.context_type == ContextType.ALL:
                context = "\n".join(self.memory[-self.max_memory :])
                user_input = f"{task.description}\n\nContext:\n{context}"
                response = assigned_agent.execute_task(user_input)

            elif self.context_type == ContextType.PREVIOUS:
                context = self.memory[-1]
                user_input = f"{task.description}\n\nContext:\n{context}"
                response = assigned_agent.execute_task(user_input)

            elif self.context_type == ContextType.SELECTED:
                context_tasks = task.context
                context = ""
                if context_tasks and len(context_tasks) > 0:
                    for context_task in context_tasks:
                        idx = self.tasks.index(context_task)
                        if idx >= len(self.memory):
                            raise IndexError(
                                f"Context task {context_task.description} not found in memory"
                            )
                        context += self.memory[idx] + "\n"
                    user_input = f"{task.description}\n\nContext:\n{context}"
                else:
                    user_input = f"{task.description}\n"
                response = assigned_agent.execute_task(user_input)

            else:
                raise ValueError(f"Invalid context_type: {self.context_type}")

            response = response.split("Final Answer:")[0].strip()
            self.memory.append(response)

        history = self.generate_history()
        print(json.dumps(history, indent=4))
        return history

    def generate_history(self):
        history = []
        for i, task in enumerate(self.tasks):
            history.append(
                {
                    "task_number": i + 1,
                    "task_description": task.description,
                    "agent_name": task.agents[0].name,
                    "response": self.memory[i],
                }
            )
        return history
