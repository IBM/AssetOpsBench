"""
Simple React-style Agent for AssetOpsBench Quickstart.
A lightweight agent that can use tools and follow basic reasoning.
"""

import re
import time
from typing import Dict, List, Any

from .base import BaseAgent


class SimpleAgent(BaseAgent):
    """A simple React-style agent for demonstration purposes."""
    
    def __init__(self, name: str, description: str, tools: List[Any]):
        super().__init__(name, description, tools)
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = 5
    
    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a task using simple React reasoning."""
        start_time = time.time()
        
        # Initialize execution context
        context = {
            'task': task,
            'observations': [],
            'thoughts': [],
            'actions': [],
            'tools_used': []
        }
        
        # Simple React loop
        for step in range(self.max_steps):
            # Think about what to do
            thought = self._think(context)
            context['thoughts'].append(thought)
            
            # Decide on action
            action = self._decide_action(thought, context)
            
            if action['type'] == 'finish':
                context['final_answer'] = action['answer']
                break
            elif action['type'] == 'use_tool':
                result = self._use_tool(action['tool'], action['input'])
                context['observations'].append(result)
                context['actions'].append(action)
                context['tools_used'].append(action['tool'])
            else:
                # Unknown action, finish
                context['final_answer'] = f"I couldn't complete the task: {task}"
                break
        
        execution_time = time.time() - start_time
        
        # Store interaction in memory
        interaction = {
            'task': task,
            'response': context.get('final_answer', 'Task incomplete'),
            'tools_used': context['tools_used'],
            'execution_time': execution_time,
            'steps': len(context['thoughts'])
        }
        self.add_to_memory(interaction)
        
        return {
            'response': context.get('final_answer', 'Task incomplete'),
            'tools_used': context['tools_used'],
            'execution_time': execution_time,
            'thoughts': context['thoughts'],
            'observations': context['observations']
        }
    
    def _think(self, context: Dict[str, Any]) -> str:
        """Generate a thought about the current situation."""
        task = context['task']
        observations = context['observations']
        
        if not observations:
            # First step - analyze the task
            if 'sites' in task.lower():
                return "The user is asking about sites. I should use the get_sites tool to retrieve available sites."
            elif 'assets' in task.lower() and 'site' in task.lower():
                return "The user is asking about assets at a specific site. I should use get_assets_by_site tool."
            elif 'assets' in task.lower() and 'type' in task.lower():
                return "The user is asking about assets by type. I should use get_assets_by_type tool."
            elif 'status' in task.lower():
                return "The user is asking about asset status. I should use get_asset_status tool."
            elif 'maintenance' in task.lower():
                return "The user is asking about maintenance. I should use get_maintenance_assets tool."
            else:
                return "I need to understand what the user is asking for and use the appropriate tool."
        else:
            # Subsequent steps - analyze observations
            last_obs = observations[-1]
            if 'error' in str(last_obs).lower():
                return f"The tool returned an error: {last_obs}. I should try a different approach or finish."
            else:
                return f"I have the information from the tool: {last_obs}. I should provide a response to the user."
    
    def _decide_action(self, thought: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what action to take based on the thought."""
        observations = context['observations']
        
        if not observations:
            # First step - try to use a tool
            task = context['task'].lower()
            
            if 'sites' in task:
                return {
                    'type': 'use_tool',
                    'tool': 'get_sites',
                    'input': {}
                }
            elif 'assets' in task and 'site' in task:
                # Extract site name from task
                site_match = re.search(r'(\w+)\s+site', context['task'], re.IGNORECASE)
                site_name = site_match.group(1) if site_match else 'MAIN'
                return {
                    'type': 'use_tool',
                    'tool': 'get_assets_by_site',
                    'input': {'site': site_name}
                }
            elif 'assets' in task and 'chiller' in task:
                return {
                    'type': 'use_tool',
                    'tool': 'get_assets_by_type',
                    'input': {'asset_type': 'chiller'}
                }
            elif 'status' in task:
                # Extract asset ID from task
                asset_match = re.search(r'(\w+-\d+)', context['task'])
                asset_id = asset_match.group(1) if asset_match else 'CH-001'
                return {
                    'type': 'use_tool',
                    'tool': 'get_asset_status',
                    'input': {'asset_id': asset_id}
                }
            elif 'maintenance' in task:
                return {
                    'type': 'use_tool',
                    'tool': 'get_maintenance_assets',
                    'input': {}
                }
            else:
                return {
                    'type': 'finish',
                    'answer': f"I'm not sure how to handle this request: {context['task']}"
                }
        else:
            # Have observations - provide final answer
            last_obs = context['observations'][-1]
            if isinstance(last_obs, dict) and 'data' in last_obs:
                if isinstance(last_obs['data'], list):
                    if len(last_obs['data']) == 0:
                        answer = "No items found."
                    else:
                        items = [str(item) for item in last_obs['data'][:5]]
                        answer = f"Found {len(last_obs['data'])} items: {', '.join(items)}"
                        if len(last_obs['data']) > 5:
                            answer += f" (and {len(last_obs['data']) - 5} more)"
                else:
                    answer = str(last_obs['data'])
            else:
                answer = str(last_obs)
            
            return {
                'type': 'finish',
                'answer': answer
            }
    
    def _use_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool with the given input."""
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            tool = self.tool_map[tool_name]
            result = tool.execute(tool_input)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
