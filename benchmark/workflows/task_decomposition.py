from agent_hive.task import Task
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from reactxen.utils.model_inference import watsonx_llm
from agent_hive.utils import json_parser
from agent_hive.logger import logger


class TaskDecompositionWorkflow:
    """
    This class represents a task decomposition workflow, where each task is decomposed into multiple subtasks
    by LLM-based task decomposition. The task decomposition is greedy, i.e., asking the LLM to generate
    the next subtask and execute it. and then do it again until the task is completely done.

    Example:
        agent1 = ...
        agent2 = ...

        task = Task(..., agents=[agent1, agent2], ...)

        workflow = PlanningWorkflow(task=[task1], ...)
        workflow.run()
    """

    llm: str = Field(description="LLM used by the task decomposition.")

    def __init__(self, tasks: List[Task], llm: str):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("TaskDecompositionWorkflow only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def run(self):
        self.memory = []
        the_task = self.tasks[0]
        history = []
        i = 0
        while True:
            response = self.decompose(the_task)

            if isinstance(response, list) and response[0] == 'respond_to_user':
                logger.info(f"Final Answer: {response[1]}")
                break

            history.append(
                {
                    'task_number': i + 1,
                    'task_description': response[1],
                    'agent_name': response[0],
                    'response': response[2]
                }
            )
            i += 1

        print(json.dumps(history, indent=4))
        return history

    def decompose(self, task: Task):
        self.memory = self.memory[-self.max_memory:]
        context = "\n".join(self.memory)
        response_format = {"action_agent": "selected_agent", "action_item": "describe_the_next_step"}
        last_response_format = {"respond_to_user": "final_answer"}

        def get_prompt():
            return f"""
You are going to solve the following task:
{task.description}

The expected output is: 
{task.expected_output}

Use the context from memory to plan next steps.                
Context:
{context}

You need will use the context provided and the user's input to classify the intent and select the appropriate agent that executes the next step.
You need to describe the next step for the selected agent so that the agent can efficiently execute it.

Here are the available agents:
{", ".join([f"- {aagent.name}" for aagent in task.agents])}
           

###Guidelines###
- The original task could require multiple steps, you will use the context to understand the previous actions taken and the next steps you should take.
- You will respond the next action in the form of {response_format}.
- If there are no actions to be taken, you will respond in the form of {last_response_format} with your final answer combining all previous responses as input.
- Respond with "respond_to_user" only when your final answer meets the expected output, or there are no agents to select from or there is no next action.
- Always return valid JSON and nothing else.                

Output:
"""

        prompt = get_prompt()
        logger.info(f"Task Decomposition Prompt: \n{prompt}")
        llm_response = watsonx_llm(prompt, model_id=self.llm)['generated_text']
        logger.info(f"Next Step: {llm_response}")

        llm_response = json_parser(llm_response)
        logger.info(f"Next Step (formatted): {llm_response}")

        if isinstance(llm_response, dict) and 'respond_to_user' in llm_response:
            return ['respond_to_user', llm_response['respond_to_user']]

        elif isinstance(llm_response, dict) and 'action_agent' in llm_response and 'action_item' in llm_response:
            action_agent = llm_response["action_agent"]
            action_item = llm_response["action_item"]

            for agent in task.agents:
                if agent.name == action_agent:
                    logger.info(f"Found agent: {agent.name}. Action item: {action_item}")
                    agent_response = agent.execute_task(action_item)
                    logger.info(f"Agent response: {agent_response}")

                    self.memory.append(agent_response)
                    return [agent.name, action_item, agent_response]
