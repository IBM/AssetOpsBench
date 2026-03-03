import json
import re
from prompt import system_prompt
from reactxen.utils.model_inference import watsonx_llm


def get_llm_answer_from_json(data: dict, model_id) -> str:
    """
    Given a parsed JSON dict with keys 'task', 'trajectory', and 'final_answer',
    formats the content and returns the LLM's response.
    """
    try:
        trajectory = data.get("trajectory", [])
        question = data.get("text", "[No question provided]")
        if len(trajectory) > 0:
            final_answer = trajectory[-1].get('final_answer', "[No final answer provided]")
        else:
            final_answer = "[No final answer provided]"

        formatted_steps = [f"Question: {question}"]
        for idx, step in enumerate(trajectory, 1):
            thought = step.get("task_description", "[No thought]")
            action = step.get("agent_name", "[No action]")
            observation = step.get("response", "[No observation]")

            step_text = (
                f"Thought {idx}: {thought}\n"
                f"Action {idx}: {action}\n"
                f"Observation {idx}: {observation}\n"
            )
            formatted_steps.append(step_text)

        formatted_steps.append(f"Answer: {final_answer}")

        # Combine all steps into a single formatted prompt
        final_prompt_string = "\n" + "-" * 40 + "\n".join(formatted_steps)
        prompt = system_prompt.format(trace=final_prompt_string)

        # Call the model inference
        # ans = watsonx_llm(prompt=prompt, model_id=16)
        ans = watsonx_llm(prompt=prompt, model_id=model_id)
        return ans

    except Exception as e:
        return f"Error while processing input data: {e}"


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and parse a JSON object from LLM-generated response text,
    even if it's wrapped in text or markdown formatting.
    """
    # Try to find a JSON block inside markdown-style code fences
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: find the first {...} block in the response
        match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            raise ValueError("No valid JSON found in the response text.")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e}")
