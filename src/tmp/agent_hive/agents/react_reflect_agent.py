import re
from typing import List, Optional

from langchain.tools import BaseTool
from reactxen.agents.react.agents import ReactReflectAgent as ReactReflectXenAgent
from reactxen.agents.react.prompts.fewshots import MPE_SIMPLE4

from agent_hive.agents.base_agent import BaseAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

class ReactReflectAgent(BaseAgent):
    """
    This class represents a ReactReflect agent that can execute a task based on user input.
    It uses a list of tools to execute the task.
    React agent can use multiple tools to execute the task.
    """
    few_shots: Optional[str] = None
    task_examples: Optional[List[str]] = None

    def __init__(self, name: str, description: str, tools: list[BaseTool], llm: str, few_shots: str = MPE_SIMPLE4,
                 task_examples: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm = llm
        self.memory = []
        self.few_shots = few_shots
        if task_examples:
            self.task_examples = task_examples
        else:
            self.task_examples = re.findall(r"^Question:(.*)$", self.few_shots, re.MULTILINE)
            self.task_examples = [ex.strip() for ex in self.task_examples]

    def execute_task(self, user_input):
        logger.info(f'ReactReflectAgent is executing task: {user_input}, with Tools {self.tools}')
        self.agent_executor = ReactReflectXenAgent(
            question=user_input,
            key="",
            cbm_tools=self.tools,
            max_steps=6,
            react_llm_model_id=self.llm,
            reflect_llm_model_id=self.llm,
            react_example=self.few_shots,
            num_reflect_iteration=5,
            handle_context_length_overflow=True,
            apply_loop_detection_check=True,
            log_structured_messages=True,
            early_stop=True,
        )
        self.agent_executor.run()
        return self.agent_executor.answer
