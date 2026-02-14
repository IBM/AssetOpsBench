"""
Mock IoT Tools for AssetOpsBench Quickstart.
Simplified versions of the IoT tools from the full benchmark.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

from .base import BaseTool


class GetSitesTool(BaseTool):
    """Mock tool to get available IoT sites."""
    
    def __init__(self):
        super().__init__(
            name="get_sites",
            description="Get all available IoT sites"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock site data."""
        sites = [
            {"id": "MAIN", "name": "Main Facility", "location": "Building A"},
            {"id": "SECONDARY", "name": "Secondary Facility", "location": "Building B"},
            {"id": "REMOTE", "name": "Remote Site", "location": "Off-site Location"}
        ]
        return {"data": sites, "count": len(sites)}


class GetAssetsBySiteTool(BaseTool):
    """Mock tool to get assets at a specific site."""
    
    def __init__(self):
        super().__init__(
            name="get_assets_by_site",
            description="Get assets located at a specific site"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock assets for the given site."""
        site = input_data.get('site', 'MAIN').upper()
        
        mock_assets = {
            "MAIN": [
                {"id": "CH-001", "name": "Chiller 1", "type": "chiller", "status": "operational"},
                {"id": "CH-002", "name": "Chiller 2", "type": "chiller", "status": "maintenance"},
                {"id": "PU-001", "name": "Pump 1", "type": "pump", "status": "operational"},
                {"id": "PU-002", "name": "Pump 2", "type": "pump", "status": "operational"}
            ],
            "SECONDARY": [
                {"id": "CH-003", "name": "Chiller 3", "type": "chiller", "status": "operational"},
                {"id": "PU-003", "name": "Pump 3", "type": "pump", "status": "maintenance"}
            ],
            "REMOTE": [
                {"id": "CH-004", "name": "Chiller 4", "type": "chiller", "status": "operational"}
            ]
        }
        
        assets = mock_assets.get(site, [])
        return {"data": assets, "count": len(assets), "site": site}


class GetAssetsByTypeTool(BaseTool):
    """Mock tool to get assets by type."""
    
    def __init__(self):
        super().__init__(
            name="get_assets_by_type",
            description="Get assets filtered by type"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock assets for the given type."""
        asset_type = input_data.get('asset_type', 'chiller').lower()
        
        mock_assets = {
            "chiller": [
                {"id": "CH-001", "name": "Chiller 1", "site": "MAIN", "status": "operational"},
                {"id": "CH-002", "name": "Chiller 2", "site": "MAIN", "status": "maintenance"},
                {"id": "CH-003", "name": "Chiller 3", "site": "SECONDARY", "status": "operational"},
                {"id": "CH-004", "name": "Chiller 4", "site": "REMOTE", "status": "operational"}
            ],
            "pump": [
                {"id": "PU-001", "name": "Pump 1", "site": "MAIN", "status": "operational"},
                {"id": "PU-002", "name": "Pump 2", "site": "MAIN", "status": "operational"},
                {"id": "PU-003", "name": "Pump 3", "site": "SECONDARY", "status": "maintenance"}
            ]
        }
        
        assets = mock_assets.get(asset_type, [])
        return {"data": assets, "count": len(assets), "type": asset_type}


class GetAssetStatusTool(BaseTool):
    """Mock tool to get status of a specific asset."""
    
    def __init__(self):
        super().__init__(
            name="get_asset_status",
            description="Get detailed status information for a specific asset"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock status for the given asset."""
        asset_id = input_data.get('asset_id', 'CH-001')
        
        mock_status = {
            "CH-001": {
                "id": "CH-001",
                "name": "Chiller 1",
                "status": "operational",
                "health_score": 95,
                "last_maintenance": "2024-01-15",
                "next_maintenance": "2024-07-15",
                "temperature": 45.2,
                "pressure": 120.5,
                "efficiency": 87.3
            },
            "CH-002": {
                "id": "CH-002",
                "name": "Chiller 2",
                "status": "maintenance",
                "health_score": 65,
                "last_maintenance": "2024-02-01",
                "next_maintenance": "2024-05-01",
                "temperature": 50.1,
                "pressure": 115.3,
                "efficiency": 72.1
            }
        }
        
        status = mock_status.get(asset_id, {
            "id": asset_id,
            "name": f"Asset {asset_id}",
            "status": "unknown",
            "health_score": 0,
            "message": "Asset not found in mock data"
        })
        
        return {"data": status}


class GetMaintenanceAssetsTool(BaseTool):
    """Mock tool to get assets that need maintenance."""
    
    def __init__(self):
        super().__init__(
            name="get_maintenance_assets",
            description="Get assets that require maintenance"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock assets needing maintenance."""
        maintenance_assets = [
            {
                "id": "CH-002",
                "name": "Chiller 2",
                "site": "MAIN",
                "priority": "high",
                "issue": "Low efficiency",
                "scheduled_date": "2024-05-01"
            },
            {
                "id": "PU-003",
                "name": "Pump 3",
                "site": "SECONDARY",
                "priority": "medium",
                "issue": "Routine maintenance",
                "scheduled_date": "2024-06-15"
            }
        ]
        
        return {"data": maintenance_assets, "count": len(maintenance_assets)}


class MockIoTTools:
    """Container for all mock IoT tools."""
    
    def __init__(self):
        self.tools = [
            GetSitesTool(),
            GetAssetsBySiteTool(),
            GetAssetsByTypeTool(),
            GetAssetStatusTool(),
            GetMaintenanceAssetsTool()
        ]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return self.tools
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool '{name}' not found")
