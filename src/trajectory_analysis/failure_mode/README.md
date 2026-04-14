# Trajectory Failure Mode Analysis

Automatically analyze LLM agent trajectories to identify and categorize failure modes using LLM-based analysis.

## 🚀 Quick Start

### Installation

```bash
# Install base dependencies
uv sync

# Optional: Install clustering dependencies
uv sync --group trajectory-analysis
```

### Basic Usage

```bash
# Analyze trajectories (default: Claude via LiteLLM)
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large

# With verbose logging
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large \
    --verbose

# With clustering enabled
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large \
    --cluster
```

### Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--path` | `-p` | `./sample_trajectories` | Input directory with trajectory JSON files |
| `--output` | `-o` | `./results` | Output directory for results |
| `--model-id` | `-m` | `litellm_proxy/aws/claude-sonnet-4-6` | LLM model for analysis |
| `--temperature` | `-t` | `0.0` | LLM temperature (0.0=deterministic) |
| `--cluster` | | `False` | Enable clustering after analysis |
| `--cluster-only` | | `False` | Skip analysis, only combine & cluster existing runs |
| `--num-clusters` | `-k` | `auto` | Number of clusters (auto-selects optimal) |
| `--verbose` | `-v` | `False` | Enable detailed logging |

## 🎯 Key Features

- **14 Predefined Failure Modes**: Automatically detects common agent failures
- **Custom Failure Discovery**: LLM identifies additional failure patterns
- **Intelligent Clustering**: Groups similar failures using embeddings
- **Multi-Format Support**: Handles different trajectory JSON structures (auto-detected)
- **Organized Output**: Timestamped runs with centralized summary
- **Timeout Protection**: 30-second timeout prevents hanging
- **Verbose Logging**: Optional detailed logging with file processing info

## 📊 Input Format

The system auto-detects and supports multiple trajectory formats:

**Format 1: Old Format (task_description/agent_name/response)**
```json
{
  "text": "Download sensor data",
  "trajectory": [
    {
      "task_description": "Connecting to database",
      "agent_name": "IoTAgent",
      "response": "Successfully connected"
    }
  ],
  "final_answer": "Data downloaded"
}
```

**Format 2: New Format (thought/action/observation)**
```json
{
  "task": "Analyze vibration data",
  "trajectory": [
    {
      "thought": "Need to retrieve sensor data",
      "action": "query_iot_database",
      "observation": "Retrieved 1000 data points"
    }
  ],
  "result": "Bearing fault detected"
}
```

## 📈 Output Structure

```
results/
├── runs/                              # Individual run results
│   ├── 20260414_140523/               # Timestamped run folder
│   │   ├── failure_modes.csv          # Analysis results (CSV)
│   │   └── failure_modes.pkl          # Analysis results (Pickle)
│   └── 20260414_153012/               # Another run
│       ├── failure_modes.csv
│       └── failure_modes.pkl
└── summary/                           # Aggregated results (created with --cluster)
    ├── combined_failure_modes.csv     # All runs combined
    ├── combined_failure_modes.pkl
    ├── additional_fm.csv              # Additional failures extracted
    └── additional_fm_clustered.csv    # Clustered results
```

### Output Files

**Individual Run (`results/runs/{timestamp}/`)**
- `failure_modes.pkl`: Pickle file with DataFrame
- `failure_modes.csv`: CSV file with same data

**Columns**:
- `model_id`: LLM model used for analysis (e.g., "litellm_proxy/claude-sonnet-4-6")
- `trajectory_path`: Relative path to trajectory file
- `counter`: Sequential row number
- `ut_id`: Trajectory identifier (e.g., "0003", "0004")
- `addi_fm_cnt`: Count of additional failure modes
- `addi_fm_list`: List of additional failure modes with descriptions
- Boolean flags for each of 14 predefined failure modes

**Summary (`results/summary/` - created with --cluster)**
- `combined_failure_modes.{pkl,csv}`: All runs combined (includes `run_id` column from folder names)
- `additional_fm.csv`: Additional failures from all runs
- `additional_fm_clustered.csv`: Clustered additional failures

## 🔍 Detected Failure Modes

### Task Execution Issues (1.x)
- **1.1** Disobey Task Specification
- **1.2** Disobey Role Specification
- **1.3** Step Repetition
- **1.4** Loss of Conversation History
- **1.5** Unaware of Termination Conditions

### Communication Issues (2.x)
- **2.1** Conversation Reset
- **2.2** Fail to Ask for Clarification
- **2.3** Task Derailment
- **2.4** Information Withholding
- **2.5** Ignored Other Agent's Input
- **2.6** Action-Reasoning Mismatch

### Verification Issues (3.x)
- **3.1** Premature Termination
- **3.2** No or Incorrect Verification
- **3.3** Weak Verification

## ⚙️ Configuration

### LiteLLM Proxy (Recommended)

```bash
export LITELLM_API_KEY="your-api-key"
export LITELLM_BASE_URL="https://your-proxy-url"

# Use with model ID
--model-id litellm_proxy/aws/claude-sonnet-4-6
--model-id litellm_proxy/gcp/gemini-2.0-flash-exp
```

### WatsonX

```bash
export WATSONX_APIKEY="your-api-key"
export WATSONX_URL="https://your-watsonx-url"
export WATSONX_PROJECT_ID="your-project-id"

# Use with model ID
--model-id watsonx/meta-llama/llama-3-3-70b-instruct
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest src/trajectory_analysis/failure_mode/tests/

# Run with coverage
uv run pytest --cov=src/trajectory_analysis/failure_mode/core
```

## 🔧 Diagnostic Tools

### Test All Available Models
```bash
python src/trajectory_analysis/failure_mode/diagnostics/test_all_litellm_models.py
```

### Test Specific Model
```bash
python src/trajectory_analysis/failure_mode/diagnostics/test_llm_model_connection.py \
    --model-id litellm_proxy/aws/claude-sonnet-4-6
```

### Verify Trajectory Format
```bash
python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py \
    --path sample_trajectories/mistral-large/0001_sample_trajectory.json \
    --show-llm-prompt
```

## 📁 Project Structure

```
failure_mode/
├── analyze_trajectories.py          # Main CLI entry point
├── core/                             # Core pipeline modules
│   ├── generator.py                  # LLM-based trajectory analysis
│   ├── reducer.py                    # Clustering and categorization
│   ├── pipeline.py                   # High-level orchestration
│   ├── utils.py                      # LLM calls and JSON parsing
│   ├── prompts.py                    # System prompts for LLM
│   ├── format_handlers.py            # Multi-format support
│   └── timeout_wrapper.py            # Timeout protection
├── diagnostics/                      # Diagnostic utilities
├── tests/                            # Test suite
├── sample_trajectories/              # Example data
└── results/                          # Output (generated)
    ├── runs/                         # Individual run results
    └── summary/                      # Combined/clustered results
```

## 💡 Usage Examples

### Example 1: Simple Analysis
```bash
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path ./trajectories/mistral-large

# Output: results/runs/20260414_140523/failure_modes.{pkl,csv}
```

### Example 2: With Clustering
```bash
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path ./trajectories/mistral-large \
    --cluster

# Output:
#   results/runs/20260414_140523/failure_modes.{pkl,csv}
#   results/summary/combined_failure_modes.{pkl,csv}
#   results/summary/additional_fm.csv
#   results/summary/additional_fm_clustered.csv
```

### Example 3: Multiple Runs, Then Cluster
```bash
# Run 1 - analyze mistral-large
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path ./trajectories/mistral-large

# Run 2 - analyze gpt4
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path ./trajectories/gpt4

# Run 3 - analyze claude
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --path ./trajectories/claude

# Now cluster all 3 runs together (no new analysis)
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --cluster-only

# Summary folder now contains combined results from all 3 runs
```

### Example 4: Cluster-Only Mode
```bash
# You already have runs in results/runs/
# Just want to re-cluster with different parameters

uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --cluster-only \
    --num-clusters 5

# Or with different embedding model
uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
    --cluster-only \
    --embedding-model all-mpnet-base-v2
```

### Example 5: Load and Analyze Results
```python
import pandas as pd

# Load individual run
df = pd.read_pickle('results/runs/20260414_140523/failure_modes.pkl')
print(df.head())

# Load combined results
combined = pd.read_pickle('results/summary/combined_failure_modes.pkl')
print(f"Total trajectories: {len(combined)}")

# Filter by failure mode
weak_verification = combined[combined['3.3 Weak Verification'] == True]
print(f"Weak verification: {len(weak_verification)} trajectories")

# Load clustered results
clusters = pd.read_csv('results/summary/additional_fm_clustered.csv')
print(clusters.groupby('cluster').size())
```

## 🐛 Troubleshooting

**"Model not available"**
- Run `test_all_litellm_models.py` to see available models
- Check API credentials

**"Timeout after 30 seconds"**
- Use a faster model or increase timeout in code

**"Invalid trajectory format"**
- Use `verify_trajectory_import.py` to check JSON structure

**"No additional failure modes found"**
- Normal if trajectories only have predefined failures
- Try with `--cluster` flag

## 📚 Additional Documentation

- **tests/README.md**: Testing strategy and test documentation
- **diagnostics/README.md**: Diagnostic tools guide

## 🤝 Contributing

1. Add tests for new features
2. Update README for new functionality
3. Follow existing code style (type hints, docstrings)
4. Run test suite before submitting

## 📄 License

See LICENSE file in repository root.
