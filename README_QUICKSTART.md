# AssetOpsBench Quickstart Mode - Pull Request #130

## 🚀 Quickstart for AssetOpsBench

This pull request adds a lightweight Quickstart / Minimal Benchmark mode for faster onboarding.

### Quick Start

```bash
# Run a quick demo
python quickstart/run_demo.py

# Run interactive mode  
python quickstart/run_quickstart.py --mode interactive

# Run batch evaluation
python quickstart/run_quickstart.py --mode batch --num-tasks 3
```

### What's Included

- ✅ **No Docker Required** - Everything runs locally
- ✅ **5-minute Setup** - Minimal dependencies
- ✅ **IoT Scenario** - Sample asset management tasks
- ✅ **Simple Agent** - React-style agent with tools
- ✅ **Local Evaluation** - Scoring and feedback system
- ✅ **Documentation** - Complete guides and examples

### Benefits

- **New Contributors**: Easy way to understand project structure
- **Researchers**: Quick experimentation without full setup
- **Students**: Educational codebase with clear examples
- **OSS Friendly**: Lower barrier to entry for contributions

### Files Added

```
quickstart/
├── README.md                 # Comprehensive documentation
├── requirements.txt          # Minimal dependencies
├── run_demo.py              # Quick demo script
├── run_quickstart.py        # Main runner with CLI
├── scenarios/               # IoT scenario implementation
├── agents/                 # Simple React agent
├── tools/                  # Mock IoT tools
├── evaluation/             # Scoring system
├── data/                  # Sample data
└── CONTRIBUTING.md         # Contribution guidelines
```

### Testing Results

- ✅ Demo mode: `python quickstart/run_demo.py` works
- ✅ Interactive mode: Chat with agent works
- ✅ Batch mode: Multiple tasks with JSON export works
- ✅ Evaluation: Scoring system provides feedback

## Issue Reference

Closes #130 - Add lightweight Quickstart / Minimal Benchmark mode for faster onboarding
