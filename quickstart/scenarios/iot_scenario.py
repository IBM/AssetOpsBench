"""
IoT Asset Management Scenario for AssetOpsBench Quickstart.
A simplified version of the IoT scenario from the full benchmark.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

from .base import BaseScenario


class IoTScenario(BaseScenario):
    """IoT Asset Management scenario for quickstart."""
    
    def __init__(self):
        super().__init__()
        self.name = "IoT Asset Management"
        self.description = "Query and manage IoT assets and sites"
        self.tasks = self._load_tasks()
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load IoT tasks from the original benchmark data."""
        # Simplified tasks based on the original iot_utterance_meta.json
        return [
            {
                "id": 1,
                "type": "IoT",
                "text": "What IoT sites are available?",
                "category": "Knowledge Query",
                "characteristic_form": "List all available IoT sites",
                "expected_tools": ["get_sites"],
                "difficulty": "easy"
            },
            {
                "id": 2,
                "type": "IoT",
                "text": "What assets can be found at the MAIN site?",
                "category": "Knowledge Query",
                "characteristic_form": "Query assets at a specific site",
                "expected_tools": ["get_assets_by_site"],
                "difficulty": "easy"
            },
            {
                "id": 3,
                "type": "IoT",
                "text": "Show me all chillers in the system",
                "category": "Knowledge Query",
                "characteristic_form": "Filter assets by type",
                "expected_tools": ["get_assets_by_type"],
                "difficulty": "medium"
            },
            {
                "id": 4,
                "type": "IoT",
                "text": "What is the current status of asset CH-001?",
                "category": "Status Query",
                "characteristic_form": "Get asset status information",
                "expected_tools": ["get_asset_status"],
                "difficulty": "medium"
            },
            {
                "id": 5,
                "type": "IoT",
                "text": "List all assets that need maintenance at the MAIN site",
                "category": "Maintenance Query",
                "characteristic_form": "Find assets requiring maintenance",
                "expected_tools": ["get_maintenance_assets"],
                "difficulty": "hard"
            }
        ]
    
    def get_task(self, task_id: int) -> Dict[str, Any]:
        """Get a specific task by ID."""
        if 1 <= task_id <= len(self.tasks):
            return self.tasks[task_id - 1]
        else:
            raise ValueError(f"Task ID {task_id} not found. Available: 1-{len(self.tasks)}")
    
    def get_sample_task(self) -> Dict[str, Any]:
        """Get a sample task for demonstration."""
        return self.tasks[0]  # Return the first task as sample
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all available tasks."""
        return self.tasks.copy()
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get tasks filtered by difficulty level."""
        return [task for task in self.tasks if task.get('difficulty') == difficulty]
    
    def get_tasks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tasks filtered by category."""
        return [task for task in self.tasks if task.get('category') == category]
