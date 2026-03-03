from reactxen.utils.model_inference import watsonx_llm
from agent_hive.agents.plan_reviewer_prompt import review_plan_system_prompt_template
import json
import re
from agent_hive.agents.base_agent import BaseAgent
from typing import List, Dict

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class SummarizationAgent(BaseAgent):
    """
    This class is responsible for summarizing the given text and answer the question.
    """

    name = "SummarizationAgent"
    description = "This agent summarizes the given text and answers the question."
    memory = []
    tools = []

    def __init__(self, llm="mistralai/mistral-large", max_retries=3):
        self.llm = llm
        self.max_retries = max_retries

    def execute_task(self, user_input):
        """
        This function execute the task by summarizing the question and provided context and then generating an answer.

        Args:
        user_input (str): The input question to be summarized and answered.

        """

        summarization_prompt = f"""Given the following context, extract the most relevant answer to the question provided. If the answer is not explicitly stated, infer it based on the context without adding external information. Output only the answer and do not include any additional text. If the question cannot be answered, respond with "Not enough information available."

Question:
{user_input}

Answer:
"""
        logger.info(f"Summarization Prompt: {summarization_prompt}")
        for it_index in range(self.max_retries):
            result = watsonx_llm(
                summarization_prompt,
                model_id=self.llm,
                temperature=it_index / 10,
                stop=["\n(END OF RESPONSE)"],
            )["generated_text"]
            if "not enough information available" not in result.lower():
                return result.strip()
            logger.warning(
                f"Retrying due to insufficient information. Attempt {it_index}/{self.max_retries}"
            )

        return ""
