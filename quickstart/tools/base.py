"""
Base tool class for AssetOpsBench Quickstart.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """Base class for all tools in the quickstart mode."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute the tool with the given input."""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data."""
        return True  # Default implementation
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        return {}  # Default implementation
