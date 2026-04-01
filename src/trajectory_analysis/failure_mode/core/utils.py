"""Utility functions for trajectory analysis and failure mode detection.

This module provides helper functions for processing trajectory data and
interacting with LLM backends for failure mode analysis.
"""

import json
import re
from typing import Dict, Any

from src.llm.base import LLMBackend
from .prompts import system_prompt
from .format_handlers import get_default_registry
from .timeout_wrapper import call_with_timeout, TimeoutError


def get_llm_answer_from_json(
    data: Dict[str, Any],
    llm_backend: LLMBackend,
    temperature: float = 0.0,
    timeout_seconds: int = 30,
) -> str:
    """
    Given a parsed JSON dict with trajectory data, formats the content and
    returns the LLM's response. Automatically detects and handles different
    trajectory formats using pluggable format handlers.

    Args:
        data: Dictionary containing trajectory data. Supports multiple formats
              through the format handler system. New formats can be added by
              registering custom handlers.
        llm_backend: LLM backend instance to use for generation
        temperature: Temperature parameter for LLM generation (default: 0.0)

    Returns:
        str: The LLM's generated response text

    Raises:
        ValueError: If no format handler can process the data
        Exception: If there's an error processing the input data
    """
    try:
        # Use format handler registry to auto-detect and parse the format
        registry = get_default_registry()
        formatted_data = registry.format_trajectory(data)

        question = formatted_data["question"]
        steps = formatted_data["steps"]
        final_answer = formatted_data["final_answer"]

        # Build formatted prompt
        formatted_steps = [f"Question: {question}"]
        for idx, step in enumerate(steps, 1):
            step_text = (
                f"Thought {idx}: {step['thought']}\n"
                f"Action {idx}: {step['action']}\n"
                f"Observation {idx}: {step['observation']}\n"
            )
            formatted_steps.append(step_text)

        formatted_steps.append(f"Answer: {final_answer}")

        # Combine all steps into a single formatted prompt
        final_prompt_string = "\n" + ("-" * 40 + "\n").join(formatted_steps)
        prompt = system_prompt.format(trace=final_prompt_string)

        # Call the LLM backend with timeout protection
        try:
            ans = call_with_timeout(
                llm_backend.generate,
                timeout_seconds=timeout_seconds,
                prompt=prompt,
                temperature=temperature,
            )
            return ans
        except TimeoutError as e:
            return f"Error: LLM call timed out after {timeout_seconds} seconds. Model may be unavailable."

    except Exception as e:
        return f"Error while processing input data: {e}"


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON object from LLM-generated response text,
    even if it's wrapped in text or markdown formatting.

    Args:
        response_text: Raw text response from LLM that may contain JSON

    Returns:
        dict: Parsed JSON object

    Raises:
        ValueError: If no valid JSON is found or JSON decoding fails
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


# Made with Bob
