from reactxen.agents.evaluation_agent.result_evaluation_prompt import system_prompt_template
from reactxen.utils.model_inference import watsonx_llm
import json
import re

class EvaluationAgent:
    """
    A class to encapsulate the logic for the EvaluationAgent, which evaluates the success or failure
    of an AI agent's response based on given criteria as well as characteristic of answer.
    """

    def __init__(self, llm=watsonx_llm, model_id=6, max_retries=3):
        """
        Initialize the EvaluationAgent.

        Args:
            llm: An instance of the language model (e.g., OpenAI, LangChain's LLM wrapper).
            model_id: Identifier for the LLM to use.
            max_retries: The maximum number of retry attempts for parsing valid JSON.
        """
        self.llm = llm
        self.model_id = model_id
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
                "suggestions": suggestions
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
            # print(f'came here : {ex}')
            # print(f'{response}')
            # Return error information if parsing fails
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

    def refine_response(
        self,
        question,
        agent_think,
        agent_response,
        error_details,
        it_index,
        review_resultFull,
        characteristic_answer,
    ):
        """
        Generate a refined prompt to request the LLM to fix JSON issues.

        Args:
            question (str): The original question or task.
            agent_think (str): The agent's explanation of its approach.
            agent_response (str): The agent's final response.
            error_details (dict): Details about the JSON decoding error.

        Returns:
            str: Refined LLM response.
        """
        refinement_prompt = (
            "Your previous response contained errors in the JSON formatting. "
            "Please ensure that your output is a valid JSON object enclosed in ```json``` blocks. "
            "Here are the error details:\n"
            f"{json.dumps(error_details, indent=2)}\n"
            "\nRegenerate your response in the requested JSON format."
        )
        prompt = system_prompt_template.format(
            question=question,
            agent_think=agent_think,
            agent_response=agent_response,
            characteristic_answer=characteristic_answer,
        )
        combined_prompt = f"{prompt}\n\nYour Response {it_index}: {review_resultFull}\n\nFeedback {it_index}: {refinement_prompt}"
        return combined_prompt

    def evaluate_response(self, question, agent_think, agent_response, characteristic_answer):
        """
        Evaluate the agent's response to a given question.

        Args:
            question (str): The original question or task.
            agent_think (str): The agent's explanation of its approach.
            agent_response (str): The agent's final response.

        Returns:
            dict: A JSON-like dictionary with the evaluation result.
        """
        prompt = system_prompt_template.format(
            question=question,
            agent_think=agent_think,
            agent_response=agent_response,
            characteristic_answer=characteristic_answer,
        )

        # Retry mechanism
        # print(f'INITIAL PROMPT = {prompt}')
        for it_index in range(self.max_retries):
            review_resultFull = self.llm(
                prompt, model_id=self.model_id, stop=["\n(END OF RESPONSE)"]
            )
            #print(review_resultFull)

            review_result = review_resultFull["generated_text"]
            parsed_result = self.extract_and_parse_json(review_result)

            # Check if parsing succeeded
            if parsed_result.get("status") != "Error":
                return parsed_result
            
            parsed_result = self.extract_and_parse_json_using_manual_parser(review_result)
            # Check if parsing succeeded
            if parsed_result.get("status") != "Error":
                return parsed_result

            # Refine response on failure
            prompt = self.refine_response(
                question,
                agent_think,
                agent_response,
                parsed_result,
                it_index,
                review_resultFull,
                characteristic_answer,
            )

        # final apply simplified

        # Return error after exceeding retries
        return {
            "status": "Error",
            "reasoning": f"Failed to produce valid JSON after {self.max_retries} attempts.",
            "suggestions": "Review the prompt and refine the LLM response strategy.",
        }
