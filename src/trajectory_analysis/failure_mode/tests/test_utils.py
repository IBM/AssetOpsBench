"""Tests for trajectory analysis utility functions.

Tests the utility functions in utils.py including JSON extraction
and LLM interaction formatting.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.trajectory_analysis.failure_mode.utils import (
    get_llm_answer_from_json,
    extract_json_from_response,
)


class TestExtractJsonFromResponse:
    """Test JSON extraction from various LLM response formats."""

    def test_extract_json_with_markdown_code_fence(self):
        """Test extraction from markdown-formatted JSON."""
        response = """Here is the analysis:
        ```json
        {
            "failure_modes": {
                "1.1 Disobey Task Specification": true,
                "1.2 Disobey Role Specification": false
            },
            "additional_failure_modes": []
        }
        ```
        That's the result.
        """
        result = extract_json_from_response(response)

        assert isinstance(result, dict)
        assert "failure_modes" in result
        assert result["failure_modes"]["1.1 Disobey Task Specification"] is True
        assert result["failure_modes"]["1.2 Disobey Role Specification"] is False

    def test_extract_json_without_code_fence(self):
        """Test extraction from plain JSON in text."""
        response = """The analysis shows: {"failure_modes": {"1.1 Disobey Task Specification": true}}"""
        result = extract_json_from_response(response)

        assert isinstance(result, dict)
        assert "failure_modes" in result
        assert result["failure_modes"]["1.1 Disobey Task Specification"] is True

    def test_extract_json_multiline_without_fence(self):
        """Test extraction from multiline JSON without code fence."""
        response = """{
            "failure_modes": {
                "1.1 Disobey Task Specification": true
            }
        }"""
        result = extract_json_from_response(response)

        assert isinstance(result, dict)
        assert "failure_modes" in result

    def test_extract_json_no_valid_json(self):
        """Test error handling when no JSON is present."""
        response = "This is just plain text with no JSON"

        with pytest.raises(ValueError, match="No valid JSON found"):
            extract_json_from_response(response)

    def test_extract_json_invalid_json_syntax(self):
        """Test error handling for malformed JSON."""
        response = """```json
        {
            "failure_modes": {
                "1.1 Disobey Task Specification": true,
            }
        }
        ```"""

        with pytest.raises(ValueError, match="JSON decoding failed"):
            extract_json_from_response(response)


class TestGetLlmAnswerFromJson:
    """Test LLM interaction and prompt formatting."""

    def test_get_llm_answer_basic_trajectory(self):
        """Test basic trajectory processing with mock LLM."""
        # Mock LLM backend
        mock_llm = Mock()
        mock_llm.generate.return_value = """```json
        {
            "failure_modes": {
                "1.1 Disobey Task Specification": true
            }
        }
        ```"""

        # Sample trajectory data
        data = {
            "text": "What is the weather?",
            "trajectory": [
                {
                    "task_description": "Check weather API",
                    "agent_name": "WeatherAgent",
                    "response": "Temperature is 72F",
                    "final_answer": "It's 72 degrees",
                }
            ],
        }

        result = get_llm_answer_from_json(data, mock_llm, temperature=0.0)

        # Verify LLM was called
        assert mock_llm.generate.called
        call_args = mock_llm.generate.call_args
        assert call_args[1]["temperature"] == 0.0

        # Verify prompt contains trajectory information
        prompt = call_args[0][0]
        assert "What is the weather?" in prompt
        assert "Check weather API" in prompt
        assert "WeatherAgent" in prompt

    def test_get_llm_answer_empty_trajectory(self):
        """Test handling of empty trajectory."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"failure_modes": {}}'

        data = {"text": "Test question", "trajectory": []}

        result = get_llm_answer_from_json(data, mock_llm)

        # Should still call LLM with formatted prompt
        assert mock_llm.generate.called
        prompt = mock_llm.generate.call_args[0][0]
        assert "Test question" in prompt
        assert "[No final answer provided]" in prompt

    def test_get_llm_answer_multiple_steps(self):
        """Test trajectory with multiple steps."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"failure_modes": {}}'

        data = {
            "text": "Multi-step task",
            "trajectory": [
                {
                    "task_description": "Step 1",
                    "agent_name": "Agent1",
                    "response": "Response 1",
                },
                {
                    "task_description": "Step 2",
                    "agent_name": "Agent2",
                    "response": "Response 2",
                    "final_answer": "Final result",
                },
            ],
        }

        result = get_llm_answer_from_json(data, mock_llm, temperature=0.5)

        prompt = mock_llm.generate.call_args[0][0]
        assert "Thought 1: Step 1" in prompt
        assert "Action 1: Agent1" in prompt
        assert "Observation 1: Response 1" in prompt
        assert "Thought 2: Step 2" in prompt
        assert "Action 2: Agent2" in prompt
        assert "Answer: Final result" in prompt

        # Verify temperature was passed
        assert mock_llm.generate.call_args[1]["temperature"] == 0.5

    def test_get_llm_answer_missing_fields(self):
        """Test handling of missing fields in trajectory steps."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"failure_modes": {}}'

        data = {
            "text": "Test",
            "trajectory": [
                {
                    # Missing task_description, agent_name, response
                }
            ],
        }

        result = get_llm_answer_from_json(data, mock_llm)

        prompt = mock_llm.generate.call_args[0][0]
        assert "[No thought]" in prompt
        assert "[No action]" in prompt
        assert "[No observation]" in prompt

    def test_get_llm_answer_error_handling(self):
        """Test error handling when LLM call fails."""
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM API error")

        data = {"text": "Test", "trajectory": []}

        result = get_llm_answer_from_json(data, mock_llm)

        # Should return error message instead of raising
        assert "Error while processing input data" in result
        assert "LLM API error" in result

    def test_get_llm_answer_temperature_default(self):
        """Test default temperature value."""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"failure_modes": {}}'

        data = {"text": "Test", "trajectory": []}

        get_llm_answer_from_json(data, mock_llm)

        # Default temperature should be 0.0
        assert mock_llm.generate.call_args[1]["temperature"] == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline_mock(self):
        """Test complete flow from data to parsed result."""
        mock_llm = Mock()
        mock_llm.generate.return_value = """```json
        {
            "failure_modes": {
                "1.1 Disobey Task Specification": true,
                "2.1 Conversation Reset": false
            },
            "additional_failure_modes": [
                {"title": "Custom Issue", "description": "A custom problem"}
            ]
        }
        ```"""

        data = {
            "text": "Complete a task",
            "trajectory": [
                {
                    "task_description": "Do something",
                    "agent_name": "TestAgent",
                    "response": "Done",
                    "final_answer": "Task completed",
                }
            ],
        }

        # Get LLM response
        llm_response = get_llm_answer_from_json(data, mock_llm)

        # Extract JSON from response
        result = extract_json_from_response(llm_response)

        # Verify complete result
        assert result["failure_modes"]["1.1 Disobey Task Specification"] is True
        assert result["failure_modes"]["2.1 Conversation Reset"] is False
        assert len(result["additional_failure_modes"]) == 1
        assert result["additional_failure_modes"][0]["title"] == "Custom Issue"


# Made with Bob
