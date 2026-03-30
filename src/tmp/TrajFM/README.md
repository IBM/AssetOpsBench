# TrajFM: LLM Trajectory Failure Mode Analysis

A pipeline for analyzing LLM agent trajectories to identify and categorize failure modes using LLM-based analysis and clustering.

> 📖 **For detailed documentation**, see [README_detail.md](README_detail.md)

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn sentence-transformers
pip install plotly kaleido  # For visualization
```

### Basic Usage

```bash
# Run the complete pipeline
python failure_mode_extractor.py \
    --traj_root_base /path/to/trajectories \
    --model_id 18 \
    --summary_dir summary_codabench

# Generate visualizations
python plot_failure_mode.py
```

### Python API

```python
from failure_mode_pipeline import run_failure_mode_pipeline

results = run_failure_mode_pipeline(
    traj_root_base="/path/to/trajectories",
    model_id=18,
    summary_dir="summary_codabench"
)
```

## 📁 Project Structure

```
TrajFM/
├── failure_mode_extractor.py      # Main CLI entry point
├── failure_mode_generator.py      # LLM-based trajectory analysis
├── failure_mode_reduction.py      # Clustering and categorization
├── failure_mode_pipeline.py       # Simplified pipeline wrapper
├── plot_failure_mode.py           # Visualization generation
├── prompt.py                       # LLM system prompt
├── utils.py                        # Helper functions
└── README_detail.md               # Detailed documentation
```

## 📊 Input Format

Trajectories should be JSON files with this structure:

```json
{
  "text": "Question or task description",
  "trajectory": [
    {
      "task_description": "What the agent is thinking",
      "agent_name": "AgentName",
      "response": "Agent's response or observation"
    }
  ],
  "final_answer": "The final answer (optional)"
}
```

## 📈 Output Files

1. **`processed_trajectories/combined_m{model_id}_db.pkl`** - Analyzed trajectories with failure modes
2. **`summary_codabench/addtional_fm.csv`** - Raw additional failure modes
3. **`summary_codabench/additional_fm_clustered.csv`** - Clustered failure modes
4. **`failure_modes_sunburst.html`** - Interactive visualization
5. **`failure_modes_sunburst.png`** - Static visualization

## 🔍 What It Detects

### 14 Predefined Failure Modes

**Task Execution Issues:**
- Disobey Task/Role Specification
- Step Repetition
- Loss of Conversation History
- Unaware of Termination Conditions

**Communication Issues:**
- Conversation Reset
- Fail to Ask for Clarification
- Task Derailment
- Information Withholding
- Ignored Other Agent's Input
- Action-Reasoning Mismatch

**Verification Issues:**
- Premature Termination
- No or Incorrect Verification
- Weak Verification

### Additional Failure Modes

The system also identifies and clusters **additional failure modes** not in the predefined list, such as:
- Inadequate Error Handling
- Lack of Final Answer
- Insufficient File Format Support
- Inconsistent Data Retrieval

## 🎨 Visualization

The `plot_failure_mode.py` script creates interactive sunburst charts showing:
- Hierarchical view of failure modes
- Cluster distributions
- Interactive drill-down capabilities

**Note:** The script includes sample data. To visualize your results, modify it to load from `summary_codabench/additional_fm_clustered.csv`.

## ⚙️ Configuration

### CLI Arguments

```bash
--traj_root_base    # Root directory containing trajectory JSON files
--model_id          # Model identifier (default: 18)
--summary_dir       # Output directory for CSV files (default: summary)
--k                 # Number of clusters (optional, auto-determined if not set)
```

### Clustering Parameters

```python
# In failure_mode_reduction.py
model_name = "all-MiniLM-L6-v2"  # Sentence transformer model
k = None  # Auto-determine optimal clusters (or set fixed number)
```

## 🔧 Customization

### Using a Different LLM Provider

Modify `utils.py` to replace the `watsonx_llm()` function with your LLM API:

```python
def get_llm_answer_from_json(data: dict, model_id: int) -> str:
    # Replace with your LLM API call
    response = your_llm_api(prompt=formatted_prompt)
    return response
```

### Adding New Predefined Failure Modes

Edit `prompt.py` to add new failure mode definitions to the `system_prompt`.

## 📝 Example Workflow

```python
# 1. Generate failure mode analysis
from failure_mode_generator import process_trajectories

gen_results = process_trajectories(
    traj_root_base="/path/to/trajectories",
    model_id=18
)

# 2. Cluster additional failure modes
from failure_mode_reduction import failure_mode_reduction

red_results = failure_mode_reduction(
    combined_db_path=gen_results['combined_path'],
    summary_dir="summary_codabench"
)

# 3. Visualize results
# Modify plot_failure_mode.py to load your CSV and run:
# python plot_failure_mode.py
```

## 🧪 Testing

```bash
# Test trajectory processing
python failure_mode_generator_test.py

# Test clustering
python failure_mode_reduction_test.py
```

## 📚 Documentation

- **[README_detail.md](README_detail.md)** - Complete documentation with:
  - Detailed function references
  - Step-by-step examples
  - Troubleshooting guide
  - Advanced customization options
  - Complete API documentation

