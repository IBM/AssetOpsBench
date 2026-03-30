from reactxen.utils.model_inference import watsonx_llm
from agent_hive.agents.plan_reviewer_prompt import review_plan_system_prompt_template
import json
import re
from agent_hive.agents.base_agent import BaseAgent
from typing import List, Dict

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class PlanReviewerAgent(BaseAgent):
    """
    This class is responsible for evaluating the generated plan based on the given criteria.
    It uses a language model to generate a review of the plan and then parses the JSON output from the model.
    """
    name = "PlanReviewerAgent"
    description = "This agent evaluates the generated plan based on predefined criteria."
    memory = []
    tools = []

    def __init__(self, llm="mistralai/mistral-large", max_retries=3):
        self.llm = llm
        self.max_retries = max_retries

    def extract_and_parse_json_using_manual_parser(self, response):

        cleaned_json_str = (
            response.strip().replace("\n", " ").replace("\\n", " ").replace("\\", "")
        )

        # Define regular expressions to extract each part:
        status_regex = r'"status":\s*"([^"]+)"'
        reasoning_regex = r'"reasoning":\s*"([^"]+)"'
        suggestions_regex = r'"suggestions":\s*"([^"]+)"'

        # Extract the values using regex
        status_match = re.search(status_regex, cleaned_json_str)
        reasoning_match = re.search(reasoning_regex, cleaned_json_str)
        suggestions_match = re.search(suggestions_regex, cleaned_json_str)

        # Extract and display the results if found
        if status_match and reasoning_match and suggestions_match:
            status = status_match.group(1)
            reasoning = reasoning_match.group(1)
            suggestions = suggestions_match.group(1)
            return {
                "status": status,
                "reasoning": reasoning,
                "suggestions": suggestions,
            }
        else:
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed.",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block.",
            }

    def extract_and_parse_json(self, response):
        """
        Extract and parse JSON from the response.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            dict: Parsed JSON object or an error report.
        """
        try:
            # Extract JSON block enclosed in ```json ... ```
            # match = re.search(r"```json(.*?)```", response, re.DOTALL)
            match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            if match:
                json_block = match.group(0).strip()  # Extract and clean the JSON block
            else:
                json_block = response.strip()

            if not json_block:
                raise ValueError("Extracted JSON block is empty.")

            parsed_json = json.loads(json_block)
            return parsed_json

        except json.JSONDecodeError as ex:
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed. {ex}",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block.",
            }

        except ValueError as ex:
            # print(f"Value Error: {ex}")
            return {
                "status": "Error",
                "reasoning": str(ex),
                "suggestions": "Check if the extracted JSON block is empty or improperly formatted.",
            }

    def execute_task(self, question: str, agent_descriptions: str, plan: str):
        """

        Evaluate the plan based on the question and agent expertise.

        Args:
            question (str): The user's question.
            agent_descriptions (str): Descriptions of the agents involved.
            plan (str): The plan to evaluate.

        Returns:
            dict: The evaluation result.
        """

        prompt = review_plan_system_prompt_template.format(
            question=question,
            agent_expertise=agent_descriptions,
            plan=plan,
        )
        logger.info(f"Review Prompt: {prompt}")
        for it_index in range(self.max_retries):
            review_result = watsonx_llm(
                prompt, model_id=self.llm, stop=["\n(END OF RESPONSE)"]
            )["generated_text"]
            # logger.info(f'review_result: {review_result}')
            parsed_result = self.extract_and_parse_json(review_result)

            # Check if parsing succeeded
            if parsed_result.get("status") != "Error":
                return parsed_result

            parsed_result = self.extract_and_parse_json_using_manual_parser(
                review_result
            )
            # Check if parsing succeeded
            if parsed_result.get("status") != "Error":
                return parsed_result

        # Return error after exceeding retries
        return {
            "status": "Error",
            "reasoning": f"Failed to produce valid JSON after {self.max_retries} attempts.",
            "suggestions": "Review the prompt and refine the LLM response strategy.",
        }
