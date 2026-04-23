"""Format handlers for different trajectory JSON structures.

This module provides a pluggable system for handling different trajectory formats.
New formats can be added by creating a new handler class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TrajectoryFormatHandler(ABC):
    """Base class for trajectory format handlers."""

    @abstractmethod
    def can_handle(self, data: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the given data format.

        Args:
            data: The trajectory data dictionary

        Returns:
            bool: True if this handler can process the data
        """
        pass

    @abstractmethod
    def extract_question(self, data: Dict[str, Any]) -> str:
        """Extract the question/task from the data."""
        pass

    @abstractmethod
    def extract_final_answer(self, data: Dict[str, Any]) -> str:
        """Extract the final answer from the data."""
        pass

    @abstractmethod
    def extract_steps(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract trajectory steps from the data.

        Returns:
            List of dicts with keys: 'thought', 'action', 'observation'
        """
        pass


class AgentResponseFormatHandler(TrajectoryFormatHandler):
    """Handler for trajectories with agent_name/task_description/response structure.

    This format uses:
    - 'text' field for the question/task
    - trajectory steps with: task_description, agent_name, response
    """

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Check if data uses agent response format (has 'text' field)."""
        return "text" in data

    def extract_question(self, data: Dict[str, Any]) -> str:
        """Extract question from 'text' field."""
        return data.get("text", "[No question provided]")

    def extract_final_answer(self, data: Dict[str, Any]) -> str:
        """Extract final answer from last trajectory step."""
        trajectory = data.get("trajectory", [])
        if trajectory:
            return trajectory[-1].get("final_answer", "[No final answer provided]")
        return "[No final answer provided]"

    def extract_steps(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract steps with task_description/agent_name/response mapping."""
        trajectory = data.get("trajectory", [])
        steps = []
        for step in trajectory:
            steps.append(
                {
                    "thought": step.get("task_description", "[No thought]"),
                    "action": step.get("agent_name", "[No action]"),
                    "observation": step.get("response", "[No observation]"),
                }
            )
        return steps


class ThoughtActionFormatHandler(TrajectoryFormatHandler):
    """Handler for trajectories with thought/action/observation structure (ReAct pattern).

    This format uses:
    - 'task' field for the question/task
    - trajectory steps with: thought, action, observation
    """

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Check if data uses thought-action format (has 'task' field)."""
        return "task" in data

    def extract_question(self, data: Dict[str, Any]) -> str:
        """Extract question from 'task' field."""
        return data.get("task", "[No question provided]")

    def extract_final_answer(self, data: Dict[str, Any]) -> str:
        """Extract final answer from root level or last trajectory step."""
        # Try root level first (new format)
        final_answer = data.get("final_answer")
        if final_answer:
            return final_answer

        # Fallback to last trajectory step
        trajectory = data.get("trajectory", [])
        if trajectory:
            return trajectory[-1].get("final_answer", "[No final answer provided]")

        return "[No final answer provided]"

    def extract_steps(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract steps with thought/action/observation fields."""
        trajectory = data.get("trajectory", [])
        steps = []
        for step in trajectory:
            steps.append(
                {
                    "thought": step.get("thought", "[No thought]"),
                    "action": step.get("action", "[No action]"),
                    "observation": step.get("observation", "[No observation]"),
                }
            )
        return steps


class TrajectoryFormatRegistry:
    """Registry for trajectory format handlers with auto-detection."""

    def __init__(self):
        """Initialize registry with default handlers."""
        self.handlers: List[TrajectoryFormatHandler] = [
            ThoughtActionFormatHandler(),  # Try thought-action format first
            AgentResponseFormatHandler(),  # Fallback to agent response format
        ]

    def register_handler(self, handler: TrajectoryFormatHandler, priority: int = 0):
        """
        Register a new format handler.

        Args:
            handler: The handler instance to register
            priority: Higher priority handlers are tried first (default: 0)
        """
        if priority > 0:
            self.handlers.insert(0, handler)
        else:
            self.handlers.append(handler)

    def get_handler(self, data: Dict[str, Any]) -> Optional[TrajectoryFormatHandler]:
        """
        Auto-detect and return the appropriate handler for the data.

        Args:
            data: The trajectory data dictionary

        Returns:
            The first handler that can process the data, or None if no handler found
        """
        for handler in self.handlers:
            if handler.can_handle(data):
                return handler
        return None

    def format_trajectory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format trajectory data into a standardized structure.

        Args:
            data: The trajectory data dictionary

        Returns:
            Standardized dict with keys: 'question', 'steps', 'final_answer', 'handler_used'

        Raises:
            ValueError: If no handler can process the data
        """
        handler = self.get_handler(data)
        if not handler:
            raise ValueError(
                f"No handler found for trajectory format. "
                f"Available keys: {list(data.keys())}"
            )

        return {
            "question": handler.extract_question(data),
            "steps": handler.extract_steps(data),
            "final_answer": handler.extract_final_answer(data),
            "handler_used": handler.__class__.__name__,
        }


# Global registry instance
_default_registry = TrajectoryFormatRegistry()


def get_default_registry() -> TrajectoryFormatRegistry:
    """Get the default global registry instance."""
    return _default_registry
