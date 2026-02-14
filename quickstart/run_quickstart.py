#!/usr/bin/env python3
"""
AssetOpsBench Quickstart Runner
Main entry point for the lightweight benchmark mode.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add quickstart to path
quickstart_path = Path(__file__).parent
sys.path.insert(0, str(quickstart_path))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(quickstart_path, '.env'))

from scenarios.iot_scenario import IoTScenario
from agents.simple_agent import SimpleAgent
from tools.mock_tools import MockIoTTools
from evaluation.simple_evaluator import SimpleEvaluator


def load_scenario(scenario_name: str):
    """Load a scenario by name."""
    scenarios = {
        'iot': IoTScenario,
    }
    
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")
    
    return scenarios[scenario_name]()


def load_agent(agent_name: str, tools: List):
    """Load an agent by name."""
    agents = {
        'simple': lambda: SimpleAgent(
            name="SimpleAgent",
            description="A simple React-style agent",
            tools=tools
        ),
    }
    
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    
    return agents[agent_name]()


def run_interactive_mode(scenario, agent, evaluator):
    """Run in interactive mode with user prompts."""
    print("🎮 Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n📝 Enter your query (or command): ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("Commands: quit, help, sample")
                continue
            elif user_input.lower() == 'sample':
                task = scenario.get_sample_task()
                user_input = task['text']
                print(f"📋 Sample task: {user_input}")
            
            if not user_input:
                continue
            
            # Execute task
            print("\n🤖 Processing...")
            start_time = time.time()
            result = agent.execute_task(user_input)
            execution_time = time.time() - start_time
            
            # Display results
            print(f"\n📤 Response: {result['response']}")
            print(f"🔧 Tools Used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
            print(f"⏱️  Time: {execution_time:.2f}s")
            
            # Simple evaluation
            mock_task = {'text': user_input, 'expected_type': 'response'}
            evaluation = evaluator.evaluate(mock_task, result)
            print(f"📊 Quality Score: {evaluation['score']}/100")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_batch_mode(scenario, agent, evaluator, num_tasks: int, output_file: str = None):
    """Run in batch mode with multiple tasks."""
    print(f"🏃‍♂️ Batch Mode - Running {num_tasks} tasks")
    print("-" * 40)
    
    results = []
    
    for i in range(num_tasks):
        task = scenario.get_task(i + 1)  # Tasks are 1-indexed
        print(f"\n📋 Task {i+1}/{num_tasks}: {task['text']}")
        
        # Execute task
        start_time = time.time()
        result = agent.execute_task(task['text'])
        execution_time = time.time() - start_time
        
        # Evaluate
        evaluation = evaluator.evaluate(task, result)
        
        # Store result
        result_data = {
            'task_id': task.get('id', i),
            'task_text': task['text'],
            'agent_response': result['response'],
            'tools_used': result['tools_used'],
            'execution_time': execution_time,
            'evaluation': evaluation
        }
        results.append(result_data)
        
        print(f"📤 Response: {result['response'][:100]}...")
        print(f"🔧 Tools: {', '.join(result['tools_used'])}")
        print(f"📊 Score: {evaluation['score']}/100")
        print(f"⏱️  Time: {execution_time:.2f}s")
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    # Summary
    avg_score = sum(r['evaluation']['score'] for r in results) / len(results)
    avg_time = sum(r['execution_time'] for r in results) / len(results)
    
    print(f"\n📈 Summary:")
    print(f"   Average Score: {avg_score:.1f}/100")
    print(f"   Average Time: {avg_time:.2f}s")
    print(f"   Total Tasks: {len(results)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AssetOpsBench Quickstart Runner')
    parser.add_argument('--scenario', default='iot', choices=['iot'],
                       help='Scenario to run (default: iot)')
    parser.add_argument('--agent', default='simple', choices=['simple'],
                       help='Agent to use (default: simple)')
    parser.add_argument('--mode', default='demo', choices=['demo', 'interactive', 'batch'],
                       help='Run mode (default: demo)')
    parser.add_argument('--num-tasks', type=int, default=3,
                       help='Number of tasks for batch mode (default: 3)')
    parser.add_argument('--output', type=str,
                       help='Output file for batch mode results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🚀 AssetOpsBench Quickstart")
    print("=" * 50)
    
    try:
        # Initialize components
        print("📦 Initializing components...")
        
        # Load tools
        tools = MockIoTTools()
        print(f"✅ Loaded {len(tools.get_all_tools())} tools")
        
        # Load agent
        agent = load_agent(args.agent, tools.get_all_tools())
        print(f"✅ Loaded agent: {agent.name}")
        
        # Load scenario
        scenario = load_scenario(args.scenario)
        print(f"✅ Loaded scenario: {scenario.name}")
        
        # Load evaluator
        evaluator = SimpleEvaluator()
        print(f"✅ Loaded evaluator")
        
        print(f"\n🎯 Running in {args.mode} mode...")
        
        if args.mode == 'demo':
            # Run a single demo task
            task = scenario.get_sample_task()
            print(f"📋 Demo task: {task['text']}")
            
            start_time = time.time()
            result = agent.execute_task(task['text'])
            execution_time = time.time() - start_time
            
            print(f"\n📤 Response: {result['response']}")
            print(f"🔧 Tools Used: {', '.join(result['tools_used'])}")
            print(f"⏱️  Execution Time: {execution_time:.2f}s")
            
            evaluation = evaluator.evaluate(task, result)
            print(f"📊 Score: {evaluation['score']}/100")
            print(f"💬 Feedback: {evaluation['feedback']}")
            
        elif args.mode == 'interactive':
            run_interactive_mode(scenario, agent, evaluator)
            
        elif args.mode == 'batch':
            run_batch_mode(scenario, agent, evaluator, args.num_tasks, args.output)
        
        print("\n✨ Completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
