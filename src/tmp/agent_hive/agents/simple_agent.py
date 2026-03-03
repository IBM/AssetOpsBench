from reactxen.utils.model_inference import watsonx_llm
from agent_hive.utils import json_parser
from langchain.tools import BaseTool
from agent_hive.agents.base_agent import BaseAgent
from agent_hive.logger import logger


class SimpleAgent(BaseAgent):
    """
    This class represents a simple agent that can execute a task based on user input.
    It uses a list of tools to execute the task.
    Only one tool is executed at a time.
    """
    def __init__(self, name: str, description: str, tools: list[BaseTool], llm: str):
        self.name = name
        self.description = description
        self.tools = tools
        self.llm = llm
        self.memory = []

    def execute_task(self, user_input):
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        response_format = {"action": "", "args": ""}

        prompt = f"""Task:
        {user_input}

        Available tools:
        {tool_descriptions}

        Based on the user's input and context, decide if you should use a tool or respond directly.        
        If you identify an action, respond with the tool name and the arguments for the tool.        
        If you decide to respond directly to the user then make the action "respond_to_user" with args as your response in the following format.

        Response Format:
        {response_format}
        
        Response:
        """
        logger.info(f"Prompt: {prompt}")
        response = watsonx_llm(prompt, model_id=self.llm, )['generated_text']
        logger.info(f"Agent Response: {response}")

        response_dict = json_parser(response)

        for tool in self.tools:
            if tool.name.lower() == response_dict["action"].lower():
                return tool.run(response_dict["args"])

        return response_dict
