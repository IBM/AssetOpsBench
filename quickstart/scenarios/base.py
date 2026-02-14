"""
Base scenario class for AssetOpsBench Quickstart.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseScenario(ABC):
    """Base class for all scenarios in the quickstart mode."""
    
    def __init__(self):
        self.name = ""
        self.description = ""
        self.tasks = []
    
    @abstractmethod
    def get_task(self, task_id: int) -> Dict[str, Any]:
        """Get a specific task by ID."""
        pass
    
    @abstractmethod
    def get_sample_task(self) -> Dict[str, Any]:
        """Get a sample task for demonstration."""
        pass
    
    @abstractmethod
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all available tasks."""
        pass
    
    def get_task_count(self) -> int:
        """Get the total number of tasks."""
        return len(self.tasks)
    
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate a task dictionary."""
        required_fields = ['id', 'text', 'type']
        return all(field in task for field in required_fields)
