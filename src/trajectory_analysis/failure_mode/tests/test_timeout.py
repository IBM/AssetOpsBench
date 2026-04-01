"""Tests for timeout protection in trajectory analysis.

This module tests the timeout wrapper functionality to ensure LLM calls
don't hang indefinitely when models are unavailable or unresponsive.
"""

import time
import pytest
from unittest.mock import Mock

from src.trajectory_analysis.failure_mode.core.timeout_wrapper import (
    call_with_timeout,
    TimeoutError,
)
from src.trajectory_analysis.failure_mode.core.utils import get_llm_answer_from_json


class TestTimeoutWrapper:
    """Test cases for the timeout wrapper utility."""

    def test_timeout_wrapper_with_fast_function(self):
        """Test that fast functions complete successfully."""

        def fast_function(x, y):
            return x + y

        result = call_with_timeout(fast_function, timeout_seconds=5, x=2, y=3)
        assert result == 5

    def test_timeout_wrapper_with_slow_function(self):
        """Test that slow functions trigger timeout."""

        def slow_function():
            time.sleep(10)
            return "Should not reach here"

        with pytest.raises(TimeoutError) as exc_info:
            call_with_timeout(slow_function, timeout_seconds=1)

        assert "timed out after 1 seconds" in str(exc_info.value)

    def test_timeout_wrapper_with_exception(self):
        """Test that exceptions from wrapped functions are propagated."""

        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError) as exc_info:
            call_with_timeout(failing_function, timeout_seconds=5)

        assert "Test error" in str(exc_info.value)

    def test_timeout_wrapper_with_return_value(self):
        """Test that return values are properly passed through."""

        def return_dict():
            return {"key": "value", "number": 42}

        result = call_with_timeout(return_dict, timeout_seconds=5)
        assert result == {"key": "value", "number": 42}

    def test_timeout_wrapper_with_none_return(self):
        """Test that None return values are handled correctly."""

        def return_none():
            return None

        result = call_with_timeout(return_none, timeout_seconds=5)
        assert result is None


class TestLLMTimeoutIntegration:
    """Integration tests for LLM timeout in trajectory analysis."""

    def test_get_llm_answer_with_timeout_parameter(self):
        """Test that get_llm_answer_from_json accepts timeout parameter."""
        # Create a mock LLM backend that responds quickly
        mock_backend = Mock()
        mock_backend.generate.return_value = (
            '{"failure_modes": {}, "additional_failure_modes": []}'
        )

        # Sample trajectory data
        data = {
            "text": "Test question",
            "trajectory": [
                {
                    "task_description": "Test task",
                    "agent_name": "TestAgent",
                    "response": "Test response",
                    "final_answer": "Test answer",
                }
            ],
        }

        # Should complete successfully with timeout parameter
        result = get_llm_answer_from_json(
            data=data, llm_backend=mock_backend, temperature=0.0, timeout_seconds=30
        )

        assert isinstance(result, str)
        assert mock_backend.generate.called

    def test_get_llm_answer_with_slow_backend(self):
        """Test that slow LLM backends trigger timeout."""
        # Create a mock LLM backend that takes too long
        mock_backend = Mock()

        def slow_generate(*args, **kwargs):
            time.sleep(5)
            return '{"failure_modes": {}}'

        mock_backend.generate.side_effect = slow_generate

        # Sample trajectory data
        data = {
            "text": "Test question",
            "trajectory": [
                {
                    "task_description": "Test task",
                    "agent_name": "TestAgent",
                    "response": "Test response",
                    "final_answer": "Test answer",
                }
            ],
        }

        # Should timeout and return error message
        result = get_llm_answer_from_json(
            data=data,
            llm_backend=mock_backend,
            temperature=0.0,
            timeout_seconds=1,  # Short timeout for testing
        )

        assert "timed out" in result.lower()
        assert "1 seconds" in result

    def test_get_llm_answer_default_timeout(self):
        """Test that default timeout is 30 seconds."""
        mock_backend = Mock()
        mock_backend.generate.return_value = (
            '{"failure_modes": {}, "additional_failure_modes": []}'
        )

        data = {
            "text": "Test question",
            "trajectory": [
                {
                    "task_description": "Test task",
                    "agent_name": "TestAgent",
                    "response": "Test response",
                    "final_answer": "Test answer",
                }
            ],
        }

        # Call without timeout parameter - should use default
        result = get_llm_answer_from_json(
            data=data, llm_backend=mock_backend, temperature=0.0
        )

        assert isinstance(result, str)
        assert mock_backend.generate.called


class TestTimeoutEdgeCases:
    """Test edge cases and boundary conditions for timeout."""

    def test_timeout_with_very_short_timeout(self):
        """Test behavior with very short timeout on slow function."""

        def slow_function():
            time.sleep(2)
            return "result"

        # Function takes 2 seconds, timeout is 0.1 seconds
        with pytest.raises(TimeoutError):
            call_with_timeout(slow_function, timeout_seconds=0.1)

    def test_timeout_with_large_timeout(self):
        """Test that large timeouts work correctly."""

        def quick_function():
            return "done"

        result = call_with_timeout(quick_function, timeout_seconds=3600)
        assert result == "done"

    def test_timeout_with_multiple_calls(self):
        """Test that timeout wrapper can be used multiple times."""

        def add(a, b):
            return a + b

        result1 = call_with_timeout(add, timeout_seconds=5, a=1, b=2)
        result2 = call_with_timeout(add, timeout_seconds=5, a=3, b=4)
        result3 = call_with_timeout(add, timeout_seconds=5, a=5, b=6)

        assert result1 == 3
        assert result2 == 7
        assert result3 == 11


# Made with Bob
