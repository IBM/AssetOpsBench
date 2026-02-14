# Contributing to AssetOpsBench Quickstart

Thank you for your interest in contributing to the AssetOpsBench Quickstart mode! This document provides guidelines for contributing to this lightweight onboarding feature.

## 🎯 Quickstart Contribution Areas

### 1. Adding New Scenarios

**Location**: `quickstart/scenarios/`

To add a new scenario:

1. **Create a new scenario file**:
```python
# quickstart/scenarios/my_scenario.py
from .base import BaseScenario

class MyScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.name = "My Custom Scenario"
        self.description = "Description of what this scenario tests"
        self.tasks = self._load_tasks()
    
    def _load_tasks(self):
        return [
            {
                "id": 1,
                "type": "Custom",
                "text": "Your task question here",
                "category": "Knowledge Query",
                "expected_tools": ["tool_name"],
                "difficulty": "easy"
            }
        ]
    
    def get_task(self, task_id: int):
        # Implementation
        pass
    
    def get_sample_task(self):
        # Implementation
        pass
    
    def get_all_tasks(self):
        # Implementation
        pass
```

2. **Register the scenario** in `quickstart/scenarios/__init__.py`:
```python
from .my_scenario import MyScenario

__all__ = ['BaseScenario', 'IoTScenario', 'MyScenario']
```

3. **Update the runner** in `quickstart/run_quickstart.py`:
```python
def load_scenario(scenario_name: str):
    scenarios = {
        'iot': IoTScenario,
        'my_scenario': MyScenario,  # Add this line
    }
    # ... rest of function
```

### 2. Adding New Agents

**Location**: `quickstart/agents/`

To add a new agent:

1. **Create a new agent file**:
```python
# quickstart/agents/my_agent.py
from .base import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, name: str, description: str, tools: List[Any]):
        super().__init__(name, description, tools)
        # Add your agent-specific initialization here
    
    def execute_task(self, task: str) -> Dict[str, Any]:
        # Implement your agent's task execution logic
        return {
            'response': 'Agent response here',
            'tools_used': ['tool1', 'tool2'],
            'execution_time': 0.5
        }
```

2. **Register the agent** in `quickstart/agents/__init__.py` and update the runner.

### 3. Adding New Tools

**Location**: `quickstart/tools/`

To add a new tool:

1. **Create a new tool file**:
```python
# quickstart/tools/my_tool.py
from .base import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Description of what this tool does"
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        # Implement your tool logic
        return {"data": "Tool result here"}
```

2. **Add to MockIoTTools** or create a new tool container class.

### 4. Improving Evaluation

**Location**: `quickstart/evaluation/`

Enhance the evaluation system by:
- Adding new scoring metrics
- Improving feedback generation
- Adding domain-specific evaluation logic

## 🧪 Testing Your Contributions

1. **Run the demo**:
```bash
python quickstart/run_demo.py
```

2. **Test your specific scenario/agent**:
```bash
python quickstart/run_quickstart.py --scenario your_scenario --agent your_agent
```

3. **Run batch tests**:
```bash
python quickstart/run_quickstart.py --mode batch --num-tasks 5 --output test_results.json
```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings to all public methods and classes
- Keep functions focused and small

### Documentation
- Update relevant README files
- Add examples for new features
- Document any configuration options

### Testing
- Ensure your contribution works with the demo
- Test edge cases and error handling
- Verify compatibility with existing components

## 🚀 Submitting Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-contribution`
3. **Make your changes** following the guidelines above
4. **Test thoroughly**
5. **Submit a pull request** with:
   - Clear description of changes
   - Testing instructions
   - Any relevant screenshots or examples

## 💡 Contribution Ideas

- **New Scenarios**: Add scenarios for different domains (manufacturing, energy, etc.)
- **Enhanced Agents**: Implement more sophisticated agent architectures
- **Better Tools**: Add more realistic mock tools with complex behavior
- **Improved Evaluation**: Develop more sophisticated evaluation metrics
- **UI/UX**: Create a simple web interface for the quickstart
- **Integration**: Add ways to easily transition from quickstart to full benchmark

## 🤝 Getting Help

- **Issues**: Report bugs or request features via GitHub issues
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check existing documentation and examples

## 📜 Quickstart Architecture

The quickstart mode follows a simple architecture:

```
quickstart/
├── scenarios/          # Task definitions and scenarios
├── agents/            # Agent implementations
├── tools/             # Mock tools for agents to use
├── evaluation/        # Scoring and feedback system
├── data/              # Sample data for mock tools
└── run_*.py          # Entry point scripts
```

Each component is designed to be:
- **Lightweight**: Minimal dependencies and fast startup
- **Extensible**: Easy to add new scenarios, agents, and tools
- **Educational**: Clear code structure for learning
- **Compatible**: Smooth transition to full benchmark

Thank you for contributing to making AssetOpsBench more accessible! 🎉
