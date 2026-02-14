"""
Base agent class for AssetOpsBench Quickstart.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseAgent(ABC):
    """Base class for all agents in the quickstart mode."""
    
    def __init__(self, name: str, description: str, tools: List[Any]):
        self.name = name
        self.description = description
        self.tools = tools
        self.memory = []
    
    @abstractmethod
    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task and return the result."""
        pass
    
    def add_to_memory(self, interaction: Dict[str, Any]):
        """Add an interaction to agent memory."""
        self.memory.append(interaction)
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get agent memory."""
        return self.memory.copy()
    
    def clear_memory(self):
        """Clear agent memory."""
        self.memory = []
