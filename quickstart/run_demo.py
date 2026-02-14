#!/usr/bin/env python3
"""
AssetOpsBench Quickstart Demo
A simple demonstration of the benchmark without Docker setup.
"""

import os
import sys
import json
from pathlib import Path

# Add quickstart to path
quickstart_path = Path(__file__).parent
sys.path.insert(0, str(quickstart_path))

from scenarios.iot_scenario import IoTScenario
from agents.simple_agent import SimpleAgent
from tools.mock_tools import MockIoTTools
from evaluation.simple_evaluator import SimpleEvaluator


def main():
    """Run a quick demo of AssetOpsBench."""
    print("🚀 AssetOpsBench Quickstart Demo")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    
    # Load mock tools
    tools = MockIoTTools()
    print(f"✅ Loaded {len(tools.get_all_tools())} mock tools")
    
    # Create simple agent
    agent = SimpleAgent(
        name="DemoAgent",
        description="A simple agent for demo purposes",
        tools=tools.get_all_tools()
    )
    print(f"✅ Created agent: {agent.name}")
    
    # Load scenario
    scenario = IoTScenario()
    print(f"✅ Loaded scenario: {scenario.name}")
    
    # Get a sample task
    task = scenario.get_sample_task()
    print(f"📋 Sample task: {task['text']}")
    print()
    
    # Execute task
    print("🤖 Executing task...")
    result = agent.execute_task(task['text'])
    
    print(f"📤 Agent Response: {result['response']}")
    print(f"🔧 Tools Used: {result['tools_used']}")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    print()
    
    # Evaluate
    print("📊 Evaluating response...")
    evaluator = SimpleEvaluator()
    evaluation = evaluator.evaluate(task, result)
    
    print(f"🎯 Score: {evaluation['score']}/100")
    print(f"💬 Feedback: {evaluation['feedback']}")
    print()
    
    print("✨ Demo completed successfully!")
    print("\n🎓 Next Steps:")
    print("1. Try: python quickstart/run_quickstart.py --help")
    print("2. Explore: quickstart/scenarios/, quickstart/agents/")
    print("3. Read: quickstart/README.md")


if __name__ == "__main__":
    main()
