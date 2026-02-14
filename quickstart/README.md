# AssetOpsBench Quickstart Mode

A lightweight, fast way to experience AssetOpsBench without Docker setup or complex dependencies.

## 🚀 Quick Start

```bash
# Install minimal dependencies
pip install -r quickstart/requirements.txt

# Run a quick demo
python quickstart/run_demo.py

# Run with custom scenario
python quickstart/run_quickstart.py --scenario iot --agent simple
```

## 📋 What's Included

The Quickstart mode provides:

- **1 Sample Scenario**: IoT asset management query
- **1 Simple Agent**: Basic React-style agent
- **Mock Data**: Pre-loaded sample asset data
- **Local Evaluation**: Simple scoring system
- **No Docker Required**: Everything runs locally

## 🎯 Learning Goals

This quickstart helps you understand:
1. How AssetOpsBench scenarios are structured
2. How agents interact with tools
3. The evaluation workflow
4. How to add your own scenarios and agents

## 📁 Quickstart Structure

```
quickstart/
├── README.md                 # This file
├── requirements.txt          # Minimal dependencies
├── run_demo.py              # Quick demo script
├── run_quickstart.py        # Main quickstart runner
├── scenarios/                # Simplified scenarios
│   ├── __init__.py
│   ├── base.py              # Base scenario class
│   └── iot_scenario.py      # IoT asset query scenario
├── agents/                  # Simple agents
│   ├── __init__.py
│   ├── base.py              # Base agent class
│   └── simple_agent.py     # Basic React agent
├── tools/                   # Mock tools
│   ├── __init__.py
│   ├── base.py              # Base tool class
│   └── mock_tools.py       # Mock IoT tools
├── data/                    # Sample data
│   └── mock_assets.json    # Sample asset data
└── evaluation/              # Simple evaluation
    ├── __init__.py
    └── simple_evaluator.py # Basic scoring
```

## 🏃‍♂️ Running Modes

### 1. Demo Mode (Fastest)
```bash
python quickstart/run_demo.py
```
Runs a single pre-configured scenario with output to console.

### 2. Interactive Mode
```bash
python quickstart/run_quickstart.py --scenario iot --agent simple --interactive
```
Run scenarios with interactive prompts.

### 3. Batch Mode
```bash
python quickstart/run_quickstart.py --scenario iot --agent simple --output results.json
```
Run multiple scenarios and save results.

## 🔧 Configuration

Create a `.env` file in the quickstart directory:
```bash
# LLM Configuration (optional - uses mock responses if not set)
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo

# Quickstart Settings
QUICKSTART_MODE=demo
MAX_SCENARIOS=3
```

## 📊 Understanding the Output

The quickstart provides:
- **Agent Response**: What the agent answered
- **Tool Usage**: Which tools were used
- **Evaluation Score**: Simple 0-100 scoring
- **Execution Time**: How long it took

## 🎓 Next Steps

After running the quickstart:

1. **Explore Full AssetOpsBench**: Try the complete benchmark with Docker
2. **Add Custom Scenarios**: Create your own scenarios in `scenarios/`
3. **Build Custom Agents**: Implement new agents in `agents/`
4. **Contribute**: Help improve the quickstart or main benchmark

## 🤝 Contributing to Quickstart

We welcome contributions! Areas to help:
- Add more sample scenarios
- Improve the simple agent
- Enhance evaluation metrics
- Better documentation

## 📚 Full Documentation

For the complete AssetOpsBench documentation, see the main README.md in the project root.
