# TrajFM: LLM Trajectory Failure Mode Analysis

A comprehensive Python pipeline for analyzing Large Language Model (LLM) execution trajectories to automatically identify, categorize, and cluster failure modes in multi-agent systems.

## 📋 Overview

This tool uses LLM-based analysis to detect 14 predefined failure modes and discover additional custom failure patterns in agent execution traces. It then applies machine learning clustering to categorize similar failure modes, providing actionable insights for debugging and quality assurance.

## 🎯 Key Features

- **Automated Failure Detection**: Analyzes trajectory JSON files using LLM to identify 14 predefined failure modes
- **Custom Failure Discovery**: LLM suggests additional failure modes specific to your system
- **Intelligent Clustering**: Groups similar failure modes using sentence embeddings and K-Means
- **Visualization**: Generate interactive sunburst and treemap charts
- **Scalable Processing**: Handles multiple timestamps and large trajectory datasets

## 📁 Project Structure

```
TrajFM/
├── failure_mode_extractor.py      # Main CLI entry point
├── failure_mode_generator.py      # Step 1: LLM-based trajectory analysis
├── failure_mode_reduction.py      # Step 2: Clustering and categorization
├── failure_mode_pipeline.py       # Simplified pipeline wrapper
├── prompt.py                       # LLM system prompt with failure mode definitions
├── utils.py                        # Helper functions for LLM calls and JSON parsing
├── plot_failure_mode.py           # Visualization generation
├── failure_mode_generator_test.py # Unit tests for generator
├── failure_mode_reduction_test.py # Unit tests for reduction
├── processed_trajectories/        # Intermediate pickle files (generated)
│   ├── 1_m12_db.pkl              # Per-timestamp analysis results
│   └── combined_m12_db.pkl       # Combined analysis across all timestamps
└── summary_codabench/             # Final output CSVs (generated)
    ├── addtional_fm.csv          # Raw additional failure modes
    └── additional_fm_clustered.csv # Clustered failure modes
```

## 📚 File Functions Reference

### Core Pipeline Files

| File | Type | Key Functions | Purpose |
|------|------|---------------|---------|
| `failure_mode_extractor.py` | **Entry Point** | `main()` | CLI interface for running the complete pipeline. Parses arguments and orchestrates generation + reduction steps. |
| `failure_mode_generator.py` | Module | `process_trajectories()`, `_load_all_json_files()`, `_normalize_additional_failure_modes()` | Loads trajectory JSON files, sends them to LLM for analysis, extracts failure modes, saves results as pickles. |
| `failure_mode_reduction.py` | Module | `failure_mode_reduction()` | Loads pickles, extracts additional failure modes, generates embeddings, performs K-Means clustering, assigns representative labels, exports CSVs. |
| `failure_mode_pipeline.py` | Wrapper | `run_failure_mode_pipeline()` | Simplified API that runs both generation and reduction steps sequentially. Returns combined results. |
| `utils.py` | Utilities | `get_llm_answer_from_json()`, `extract_json_from_response()` | Formats trajectory data for LLM prompts, calls watsonx API, parses JSON responses (handles markdown formatting). |
| `prompt.py` | Configuration | `system_prompt` (string) | Contains the complete system prompt with failure mode definitions, examples, and JSON output format instructions for the LLM. |
| `plot_failure_mode.py` | Visualization | Script execution | Generates interactive hierarchical visualizations (sunburst charts) using Plotly to display failure mode distributions across clusters. |

### Test Files

| File | Purpose |
|------|---------|
| `failure_mode_generator_test.py` | Unit tests for trajectory processing and LLM analysis functions |
| `failure_mode_reduction_test.py` | Unit tests for clustering and reduction algorithms |

### Function Details

#### `failure_mode_generator.py`

```python
def process_trajectories(
    timestamps: Optional[Sequence[str]] = None,
    traj_root_base: str = ".",
    model_id: int = 18,
    out_dir: str = "processed_trajectories"
) -> Dict[str, Any]
```
**Returns**: Dictionary with keys:
- `per_timestamp_paths`: List of individual pickle file paths
- `combined_path`: Path to combined pickle file
- `combined_df`: Pandas DataFrame with all results

**Process**:
1. Auto-discovers subdirectories if `timestamps=None`
2. Loads all JSON files recursively using `_load_all_json_files()`
3. For each trajectory:
   - Formats as prompt using `get_llm_answer_from_json()`
   - Sends to LLM for analysis
   - Extracts failure modes from JSON response
   - Stores in DataFrame row
4. Saves per-timestamp and combined pickle files

```python
def _load_all_json_files(root_path: str) -> Dict[str, Any]
```
**Returns**: Dictionary mapping file paths to parsed JSON content
**Process**: Recursively walks directory tree, attempts to load each file as JSON

```python
def _normalize_additional_failure_modes(obj: Any) -> List[Dict[str, Any]]
```
**Returns**: List of dicts with `title` and `description` keys
**Process**: Handles various formats (list, dict, single object) and normalizes to standard format

#### `failure_mode_reduction.py`

```python
def failure_mode_reduction(
    combined_pickle_path: str,
    out_dir: str = "summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 7,
    verbose: bool = True
) -> Dict[str, Any]
```
**Returns**: Dictionary with keys:
- `df_expanded`: Tidy DataFrame with title/description columns
- `df_clustered`: Clustered DataFrame with cluster/failure mode labels
- `k`: Number of clusters used
- `silhouette_scores`: List of (k, score) tuples if auto-selected
- `paths`: Dictionary with CSV file paths

**Process**:
1. Loads combined pickle file
2. Explodes `addi_fm_list` column into separate rows
3. Normalizes to title/description format
4. Saves raw CSV (`addtional_fm.csv`)
5. Generates sentence embeddings using SentenceTransformer
6. If `k=None`: Tests k_min to k_max clusters, selects best via silhouette score
7. Performs K-Means clustering
8. Assigns representative title (closest to centroid) to each cluster
9. Saves clustered CSV (`additional_fm_clustered.csv`)

#### `utils.py`

```python
def get_llm_answer_from_json(data: dict, model_id: int) -> str
```
**Parameters**:
- `data`: Dict with keys `text` (question), `trajectory` (list of steps)
- `model_id`: LLM model identifier

**Returns**: Raw LLM response text

**Process**:
1. Extracts question, trajectory steps, and final answer
2. Formats each step as "Thought X / Action X / Observation X"
3. Combines into formatted trace string
4. Inserts into system prompt template
5. Calls `watsonx_llm()` API
6. Returns response

```python
def extract_json_from_response(response_text: str) -> dict
```
**Returns**: Parsed JSON dictionary

**Process**:
1. Searches for JSON in markdown code fences (```json...```)
2. Falls back to finding first {...} block
3. Parses and returns JSON object
4. Raises ValueError if no valid JSON found

#### `failure_mode_pipeline.py`

```python
def run_failure_mode_pipeline(
    traj_root_base: str,
    model_id: int = 18,
    timestamps: Optional[Sequence[str]] = None,
    summary_dir: str = "summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: Optional[int] = None
) -> Dict[str, Any]
```
**Returns**: Dictionary with keys:
- `generation`: Results from `process_trajectories()`
- `reduction`: Results from `failure_mode_reduction()`

**Process**:
1. Calls `process_trajectories()` with provided parameters
2. Calls `failure_mode_reduction()` using combined pickle from step 1
3. Returns both result dictionaries

#### `plot_failure_mode.py`

**Type**: Standalone visualization script (not imported as module)

**Purpose**: Creates interactive hierarchical visualizations of failure mode distributions using Plotly sunburst charts.

**Data Structure**:
The script contains hardcoded sample data with three hierarchical levels:
1. **Cluster** (0-6): Top-level grouping of related failure modes
2. **Failure Mode**: General category (e.g., "Inadequate Error Handling", "Lack of Final Answer")
3. **Title**: Specific failure mode instance with detailed description

**Sample Data Included**:
- 24 failure mode examples across 7 clusters
- Categories include:
  - Inadequate Error Handling
  - Insufficient File Format Support
  - Lack of Final Answer
  - Lack of Adaptive Learning
  - Inconsistent Data Retrieval
  - Insufficient Feedback
  - Inadequate Finalization

**Outputs**:
1. `failure_modes_sunburst.html` - Interactive HTML visualization with hover details
2. `failure_modes_sunburst.png` - Static high-resolution PNG (scale=2)

**Visualization Features**:
- Color-coded by cluster using Viridis color scale
- Shows label + percentage for each segment
- Hover displays: label, value, and percentage of parent
- Interactive drill-down through hierarchy levels

**Prerequisites**:
```bash
pip install plotly kaleido  # kaleido required for PNG export
```

**Usage**:
```bash
python plot_failure_mode.py
```

**Note**: This script uses **sample/demonstration data** hardcoded in the file. To visualize your own pipeline results, you would need to:
1. Load data from `summary_codabench/additional_fm_clustered.csv`
2. Transform it into the required format (cluster, failure_mode, title columns)
3. Replace the `data` dictionary in the script

**Customization Example**:
```python
import pandas as pd
import plotly.express as px

# Load your clustered results
df = pd.read_csv('summary_codabench/additional_fm_clustered.csv')

# Create sunburst (adjust path based on your column names)
fig = px.sunburst(
    df,
    path=['cluster', 'failure_mode', 'title'],
    color='cluster',
    color_continuous_scale='Viridis',
    title="Your Failure Modes Analysis"
)
fig.write_html("your_results.html")
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn sentence-transformers plotly kaleido
```

**Note**: You'll need access to IBM watsonx or modify `utils.py` to use your preferred LLM API.

### Entry Points

There are **three ways** to run the pipeline:

#### 1. **Main Entry Point: Command Line Interface** (Recommended)

The primary entry point is `failure_mode_extractor.py`, which provides a complete CLI:

```bash
python failure_mode_extractor.py \
    --traj_directory ./path/to/trajectories \
    --model_id 18 \
    --summary_dir summary \
    --model_name all-MiniLM-L6-v2 \
    --k 5
```

**What it does**:
1. Reads all JSON trajectory files from `--traj_directory`
2. Analyzes each trajectory using LLM (model specified by `--model_id`)
3. Saves intermediate results to `processed_trajectories/`
4. Clusters additional failure modes
5. Exports final CSVs to `--summary_dir`

#### 2. **Python API: Pipeline Wrapper**

For programmatic access, use `failure_mode_pipeline.py`:

```python
from failure_mode_pipeline import run_failure_mode_pipeline

results = run_failure_mode_pipeline(
    traj_root_base="./path/to/trajectories",
    model_id=18,
    timestamps=None,  # Auto-discover subdirectories
    summary_dir="summary",
    model_name="all-MiniLM-L6-v2",
    k=None  # Auto-select optimal K using silhouette scores
)

# Access generation results
print(f"Analyzed {len(results['generation']['combined_df'])} trajectories")
print(results["generation"]["combined_df"].head())

# Access clustering results
print(f"Found {results['reduction']['k']} clusters")
print(results["reduction"]["df_clustered"].head())
```

#### 3. **Step-by-Step: Individual Modules**

For fine-grained control, call modules separately:

```python
# Step 1: Generate failure mode analysis
from failure_mode_generator import process_trajectories

gen_results = process_trajectories(
    traj_root_base="./my_trajectories",
    model_id=18,
    out_dir="processed_trajectories"
)

# Step 2: Cluster additional failure modes
from failure_mode_reduction import failure_mode_reduction

red_results = failure_mode_reduction(
    combined_pickle_path=gen_results["combined_path"],
    out_dir="summary",
    k=None  # Auto-select
)
```

### How Trajectories Are Read

The pipeline automatically discovers and reads trajectory files:

```python
# In failure_mode_generator.py
def _load_all_json_files(root_path: str) -> Dict[str, Any]:
    """
    Recursively walks the directory tree starting from root_path.
    Attempts to load each file as JSON.
    Returns a dictionary mapping file_path -> parsed_json_content
    """
```

**Directory Structure Expected**:
```
traj_root_base/
├── trajectory_file_1.json
├── trajectory_file_2.json
├── subdirectory/
│   ├── trajectory_file_3.json
│   └── trajectory_file_4.json
└── another_subdir/
    └── trajectory_file_5.json
```

**Reading Process**:
1. `os.walk()` recursively traverses all subdirectories
2. Each file is opened and parsed as JSON
3. Files that fail JSON parsing are silently skipped
4. File path is used to extract metadata (e.g., `ut_id` from filename)
5. All successfully loaded trajectories are processed

**Filename Convention**:
- Files are typically named like: `{ut_id}_{other_info}.json`
- The `ut_id` (unit test ID) is extracted from the filename using: `parts = os.path.relpath(path, root_directory).split('_')`

## 📊 Input Format

### Trajectory JSON Structure

Your trajectory files **must** be JSON with the following structure:

```json
{
  "text": "User's question or task description",
  "trajectory": [
    {
      "task_description": "Agent's thought process",
      "agent_name": "Action taken",
      "response": "Observation or result"
    }
  ]
}
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `text` | string | The original user question or task | `"Download sensor data for equipment CWC04009"` |
| `trajectory` | array | List of execution steps (can be empty) | `[{...}, {...}]` |
| `trajectory[].task_description` | string | Agent's reasoning or thought | `"I need to retrieve the sensor data"` |
| `trajectory[].agent_name` | string | Name of agent or action taken | `"IoTDataDownload"` |
| `trajectory[].response` | string | Result or observation from action | `"Data downloaded successfully"` |

### Complete Example

```json
{
  "text": "Download sensor data for equipment CWC04009 from 2020",
  "trajectory": [
    {
      "task_description": "I need to retrieve the sensor data for the specified equipment",
      "agent_name": "IoTDataDownload",
      "response": "Successfully connected to IoT database"
    },
    {
      "task_description": "Now I'll query the data for year 2020",
      "agent_name": "DatabaseQuery",
      "response": "Retrieved 1000 data points"
    },
    {
      "task_description": "I should analyze the data format",
      "agent_name": "DataAnalyzer",
      "response": "Error: Invalid file format - expected CSV but got JSON"
    },
    {
      "task_description": "Let me try to convert the format",
      "agent_name": "FormatConverter",
      "response": "Conversion failed - missing required fields"
    }
  ]
}
```

### How Trajectories Are Processed

When a trajectory is loaded, it's formatted into a prompt like this:

```
Question: Download sensor data for equipment CWC04009 from 2020
----------------------------------------
Thought 1: I need to retrieve the sensor data for the specified equipment
Action 1: IoTDataDownload
Observation 1: Successfully connected to IoT database
----------------------------------------
Thought 2: Now I'll query the data for year 2020
Action 2: DatabaseQuery
Observation 2: Retrieved 1000 data points
----------------------------------------
Thought 3: I should analyze the data format
Action 3: DataAnalyzer
Observation 3: Error: Invalid file format - expected CSV but got JSON
----------------------------------------
Thought 4: Let me try to convert the format
Action 4: FormatConverter
Observation 4: Conversion failed - missing required fields
----------------------------------------
Answer: [No final answer provided]
```

This formatted trace is then sent to the LLM along with the system prompt for failure mode analysis.

### Edge Cases Handled

- **Empty trajectory**: If `trajectory` is an empty list, only the question is analyzed
- **Missing final_answer**: Extracted from last trajectory step if available, otherwise marked as `"[No final answer provided]"`
- **Missing fields**: Default values used (`"[No thought]"`, `"[No action]"`, `"[No observation]"`)

## 🔍 Predefined Failure Modes

The system detects 14 predefined failure modes across three categories:

### Task Execution Issues (1.x)
- **1.1 Disobey Task Specification**: Fails to follow task constraints
- **1.2 Disobey Role Specification**: Violates assigned role responsibilities
- **1.3 Step Repetition**: Unnecessarily repeats completed tasks
- **1.4 Loss of Conversation History**: Loses critical context
- **1.5 Unaware of Termination Conditions**: Doesn't recognize when to stop

### Communication Issues (2.x)
- **2.1 Conversation Reset**: Unexpectedly restarts dialogue
- **2.2 Fail to Ask for Clarification**: Proceeds with unclear information
- **2.3 Task Derailment**: Deviates from intended objective
- **2.4 Information Withholding**: Doesn't share critical information
- **2.5 Ignored Other Agent's Input**: Disregards suggestions
- **2.6 Action-Reasoning Mismatch**: Actions contradict reasoning

### Verification Issues (3.x)
- **3.1 Premature Termination**: Ends before completion
- **3.2 No or Incorrect Verification**: Fails to verify outputs
- **3.3 Weak Verification**: Insufficient verification rigor

## 📈 Output Files

### 1. `processed_trajectories/combined_m{model_id}_db.pkl`
Pandas DataFrame with columns:
- `model_id`, `counter`, `timestamp`, `vendor`, `model`, `ut_id`
- Boolean columns for each of the 14 predefined failure modes
- `addi_fm_cnt`: Count of additional failure modes
- `addi_fm_list`: List of dicts with `title` and `description`

### 2. `summary_codabench/addtional_fm.csv`
Raw additional failure modes:
```csv
title,description
Inadequate Error Handling,"System failed to handle incorrect file format..."
Insufficient Feedback,"No feedback provided about task outcome..."
```

### 3. `summary_codabench/additional_fm_clustered.csv`
Clustered failure modes:
```csv
cluster,failure mode,title,description
1,Inadequate Error Handling,Inadequate Error Handling,"System failed..."
1,Inadequate Error Handling,Lack of Error Handling,"Did not handle..."
2,Lack of Final Answer,Lack of Final Answer,"Failed to provide..."
```

## 🎨 Visualization

The project includes a visualization script to create interactive hierarchical charts of failure mode distributions.

### Quick Start

```bash
python plot_failure_mode.py
```

### What It Does

Creates a **sunburst chart** - a circular hierarchical visualization with three levels:
1. **Inner ring**: Specific failure mode titles (most detailed)
2. **Middle ring**: General failure mode categories
3. **Outer ring**: Cluster groupings (color-coded)

### Output Files

- **`failure_modes_sunburst.html`** - Interactive HTML visualization
  - Hover over segments to see details
  - Click to drill down into hierarchy
  - Shows percentages and counts
  
- **`failure_modes_sunburst.png`** - Static high-resolution image (2x scale)
  - Suitable for papers and presentations
  - Requires `kaleido` package

### Prerequisites

```bash
pip install plotly kaleido
```

### Sample Data

The script includes **demonstration data** with 24 failure mode examples across 7 clusters, including:
- Inadequate Error Handling
- Lack of Final Answer
- Insufficient File Format Support
- Lack of Adaptive Learning
- Inconsistent Data Retrieval
- Insufficient Feedback
- Inadequate Finalization

### Using Your Own Data

To visualize results from your pipeline:

```python
import pandas as pd
import plotly.express as px

# Load your clustered results
df = pd.read_csv('summary_codabench/additional_fm_clustered.csv')

# Ensure you have these columns: cluster, failure_mode, title
# You may need to rename or create them based on your CSV structure

fig = px.sunburst(
    df,
    path=['cluster', 'failure_mode', 'title'],  # hierarchy levels
    color='cluster',
    color_continuous_scale='Viridis',
    title="Failure Modes Analysis - Your Results"
)

fig.update_traces(textinfo='label+percent entry', hoverinfo='label+value+percent parent')
fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))

fig.write_html("my_failure_modes.html")
fig.write_image("my_failure_modes.png", scale=2)
```

### Customization Options

**Color Schemes**:
```python
color_continuous_scale='Viridis'  # Default
color_continuous_scale='Plasma'
color_continuous_scale='Inferno'
color_continuous_scale='RdYlBu'
```

**Chart Size**:
```python
fig.update_layout(width=1000, height=1000)  # Adjust dimensions
```

**Export Quality**:
```python
fig.write_image("output.png", scale=3)  # Higher resolution (3x)
```

### Troubleshooting

**Error: "kaleido not found"**
```bash
pip install kaleido
```

**Error: "No module named 'plotly'"**
```bash
pip install plotly
```

**Empty or incorrect visualization**:
- Verify your CSV has the required columns: `cluster`, `failure_mode`, `title`
- Check that cluster values are numeric or can be converted to colors
- Ensure there are no null values in the hierarchy columns

## ⚙️ Configuration Options

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--traj_directory` | str | `./localtemp/trajectory/` | Root directory with trajectory JSON files |
| `--model_id` | int | `18` | LLM model ID for analysis |
| `--summary_dir` | str | `summary` | Output directory for CSV files |
| `--model_name` | str | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `--k` | int | `None` | Fixed cluster count (None = auto-select) |
| `--timestamps` | list | `None` | Specific timestamps to process (None = all) |

### Clustering Parameters

In `failure_mode_reduction()`:
- `k_min`: Minimum clusters to try (default: 2)
- `k_max`: Maximum clusters to try (default: 7)
- Optimal K selected using silhouette score analysis

## 🔧 Customization for Your Research

### Critical Modifications Required

#### 1. **Entry Point (`failure_mode_extractor.py`)**

**What to Modify**: Command-line arguments when running

**Current Usage**:
```bash
python failure_mode_extractor.py \
    --traj_directory ./localtemp/trajectory/ \
    --model_id 18
```

**Your Modifications**:

| Argument | Default | What to Change | Example |
|----------|---------|----------------|---------|
| `--traj_directory` | `./localtemp/trajectory/` | **REQUIRED**: Path to your trajectory JSON files | `--traj_directory /data/my_trajectories` |
| `--model_id` | `18` | **OPTIONAL**: Your LLM model ID | `--model_id 20` |
| `--summary_dir` | `summary` | **OPTIONAL**: Output directory name | `--summary_dir my_results` |
| `--model_name` | `all-MiniLM-L6-v2` | **OPTIONAL**: Embedding model | `--model_name paraphrase-MiniLM-L6-v2` |
| `--k` | `None` (auto) | **OPTIONAL**: Fixed cluster count | `--k 5` |
| `--timestamps` | `None` (all) | **OPTIONAL**: Specific timestamps | `--timestamps 2024-01 2024-02` |

**Example for Your Research**:
```bash
python failure_mode_extractor.py \
    --traj_directory /home/researcher/experiment_data/trajectories \
    --model_id 18 \
    --summary_dir results_experiment1 \
    --k 6
```

**No Code Changes Needed** - Just modify command-line arguments!

#### 2. **Using a Different LLM Provider**

**What to Modify**: `utils.py` file (lines 40-41)

**Current Code**:
```python
# ans = watsonx_llm(prompt=prompt, model_id=16)
ans = watsonx_llm(prompt=prompt, model_id=model_id)
```

**Modification Options**:

**Option A: OpenAI GPT**
```python
import openai

def get_llm_answer_from_json(data: dict, model_id) -> str:
    # ... existing formatting code ...
    
    # Replace watsonx_llm with OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    ans = {
        "generated_text": response.choices[0].message.content
    }
    return ans
```

**Option B: Anthropic Claude**
```python
import anthropic

def get_llm_answer_from_json(data: dict, model_id) -> str:
    # ... existing formatting code ...
    
    client = anthropic.Anthropic(api_key="your-api-key")
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    ans = {
        "generated_text": message.content[0].text
    }
    return ans
```

**Option C: Local Model (Ollama)**
```python
import requests

def get_llm_answer_from_json(data: dict, model_id) -> str:
    # ... existing formatting code ...
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",  # or your local model
            "prompt": prompt,
            "stream": False
        }
    )
    ans = {
        "generated_text": response.json()["response"]
    }
    return ans
```

**Option D: Azure OpenAI**
```python
from openai import AzureOpenAI

def get_llm_answer_from_json(data: dict, model_id) -> str:
    # ... existing formatting code ...
    
    client = AzureOpenAI(
        api_key="your-api-key",
        api_version="2024-02-01",
        azure_endpoint="https://your-resource.openai.azure.com"
    )
    response = client.chat.completions.create(
        model="your-deployment-name",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    ans = {
        "generated_text": response.choices[0].message.content
    }
    return ans
```

#### 3. **Test Files Modifications**

See the [Testing section](#-testing) above for detailed modification instructions for:
- `failure_mode_generator_test.py` - Change `traj_root_base` path
- `failure_mode_reduction_test.py` - Update pickle path and parameters

### Adding New Predefined Failure Modes

1. Update `failure_mode_keys` list in `failure_mode_generator.py`
2. Add definitions to `prompt.py` system prompt
3. Update DataFrame columns accordingly

### Customizing Clustering

Adjust parameters in `failure_mode_reduction.py`:
```python
# Change embedding model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Modify clustering algorithm
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=2)
```

## 📝 Complete Example Workflow

### End-to-End Example

```python
# 1. Analyze trajectories
from failure_mode_generator import process_trajectories

gen_results = process_trajectories(
    traj_root_base="./my_trajectories",
    model_id=18,
    out_dir="processed_trajectories"
)

print(f"Analyzed {len(gen_results['combined_df'])} trajectories")
print(f"Found {gen_results['combined_df']['addi_fm_cnt'].sum()} additional failure modes")

# 2. Cluster failure modes
from failure_mode_reduction import failure_mode_reduction

red_results = failure_mode_reduction(
    combined_pickle_path=gen_results["combined_path"],
    out_dir="summary",
    k=None  # Auto-select
)

print(f"Optimal K: {red_results['k']}")
print(f"Silhouette scores: {red_results['silhouette_scores']}")

# 3. Analyze results
df = red_results['df_clustered']
print("\nFailure Mode Distribution:")
print(df['failure mode'].value_counts())

# 4. Generate visualizations
import subprocess
subprocess.run(["python", "plot_failure_mode.py"])
```

### Detailed Workflow with Data Inspection

```python
import pandas as pd
from failure_mode_pipeline import run_failure_mode_pipeline

# Run complete pipeline
results = run_failure_mode_pipeline(
    traj_root_base="./trajectories",
    model_id=18,
    summary_dir="summary",
    k=None  # Auto-select optimal K
)

# === Inspect Generation Results ===
gen_df = results["generation"]["combined_df"]

print("=== GENERATION RESULTS ===")
print(f"Total trajectories analyzed: {len(gen_df)}")
print(f"\nColumns: {list(gen_df.columns)}")

# Check predefined failure modes
predefined_cols = [col for col in gen_df.columns if col.startswith(('1.', '2.', '3.'))]
print(f"\nPredefined failure mode occurrences:")
for col in predefined_cols:
    count = gen_df[col].sum()
    if count > 0:
        print(f"  {col}: {count} ({count/len(gen_df)*100:.1f}%)")

# Check additional failure modes
print(f"\nAdditional failure modes:")
print(f"  Total count: {gen_df['addi_fm_cnt'].sum()}")
print(f"  Trajectories with additional FMs: {(gen_df['addi_fm_cnt'] > 0).sum()}")
print(f"  Average per trajectory: {gen_df['addi_fm_cnt'].mean():.2f}")

# === Inspect Reduction Results ===
red_df = results["reduction"]["df_clustered"]

print("\n=== CLUSTERING RESULTS ===")
print(f"Number of clusters: {results['reduction']['k']}")
print(f"Total additional failure modes: {len(red_df)}")

# Cluster distribution
print("\nCluster distribution:")
cluster_counts = red_df.groupby('failure mode').size().sort_values(ascending=False)
for fm, count in cluster_counts.items():
    print(f"  {fm}: {count} instances")

# Show sample from each cluster
print("\n=== SAMPLE FROM EACH CLUSTER ===")
for cluster_id in sorted(red_df['cluster'].unique()):
    cluster_data = red_df[red_df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}: {cluster_data.iloc[0]['failure mode']}")
    print(f"  Sample title: {cluster_data.iloc[0]['title']}")
    print(f"  Sample description: {cluster_data.iloc[0]['description'][:100]}...")

# === Export for Further Analysis ===
# Save detailed report
with open("analysis_report.txt", "w") as f:
    f.write("=== FAILURE MODE ANALYSIS REPORT ===\n\n")
    f.write(f"Total Trajectories: {len(gen_df)}\n")
    f.write(f"Trajectories with Failures: {(gen_df[predefined_cols].any(axis=1)).sum()}\n")
    f.write(f"Additional Failure Modes: {gen_df['addi_fm_cnt'].sum()}\n")
    f.write(f"Unique Failure Clusters: {results['reduction']['k']}\n\n")
    
    f.write("Top Predefined Failure Modes:\n")
    for col in predefined_cols:
        count = gen_df[col].sum()
        if count > 0:
            f.write(f"  {col}: {count}\n")
    
    f.write("\nClustered Additional Failure Modes:\n")
    for fm, count in cluster_counts.items():
        f.write(f"  {fm}: {count}\n")

print("\nReport saved to analysis_report.txt")
```

### Reading Existing Results

If you've already run the pipeline and want to analyze existing results:

```python
import pandas as pd

# Load generation results
gen_df = pd.read_pickle("processed_trajectories/combined_m18_db.pkl")

# Load clustering results
clustered_df = pd.read_csv("summary/additional_fm_clustered.csv")
raw_df = pd.read_csv("summary/addtional_fm.csv")

# Analyze
print(f"Trajectories: {len(gen_df)}")
print(f"Additional FMs: {len(raw_df)}")
print(f"Clusters: {clustered_df['cluster'].nunique()}")

# Find trajectories with specific failure mode
step_repetition = gen_df[gen_df['1.3 Step Repetition'] == True]
print(f"\nTrajectories with Step Repetition: {len(step_repetition)}")
print(step_repetition[['ut_id', 'addi_fm_cnt']].head())
```

## 🧪 Testing

### Test Files Overview

The repository includes two test files that demonstrate basic usage:

#### 1. `failure_mode_generator_test.py`

**Purpose**: Tests the trajectory generation/analysis step

**Current Code**:
```python
from failure_mode_generator import process_trajectories

res = process_trajectories(
    traj_root_base="/Users/dhaval/Documents/GitHub/CodeBenchBackEnd/codabench/tmp0k8_i0of/trajectory",
    model_id=12,
)

print(res)
```

**What You Need to Modify**:

| Parameter | Current Value | What to Change |
|-----------|---------------|----------------|
| `traj_root_base` | `/Users/dhaval/Documents/.../trajectory` | **REQUIRED**: Change to your trajectory directory path |
| `model_id` | `12` | **OPTIONAL**: Change to your LLM model ID (default: 18) |

**Example Modifications**:

```python
# For your own research
from failure_mode_generator import process_trajectories

res = process_trajectories(
    traj_root_base="./my_research_data/trajectories",  # ← Your path
    model_id=18,  # ← Your model ID (or keep default)
    out_dir="processed_trajectories",  # ← Optional: change output directory
)

print(f"Processed {len(res['combined_df'])} trajectories")
print(f"Saved to: {res['combined_path']}")
print(res['combined_df'].head())
```

**To Run**:
```bash
# After modifying the path
python failure_mode_generator_test.py
```

#### 2. `failure_mode_reduction_test.py`

**Purpose**: Tests the clustering/reduction step

**Current Code**:
```python
from failure_mode_reduction import failure_mode_reduction

result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m12_db.pkl",
    out_dir="summary_codabench",
    # model_name="all-MiniLM-L6-v2",  # or another sentence-transformers model
    # k=6,                             # fix cluster count if you prefer
)

print(result["k"], result["silhouette_scores"][:3])
print(result["paths"])
print(result["df_clustered"].head())
```

**What You Need to Modify**:

| Parameter | Current Value | What to Change |
|-----------|---------------|----------------|
| `combined_pickle_path` | `./processed_trajectories/combined_m12_db.pkl` | **REQUIRED**: Update model ID (m12 → m18) or path if you changed it |
| `out_dir` | `summary_codabench` | **OPTIONAL**: Change output directory name |
| `model_name` | (commented) `all-MiniLM-L6-v2` | **OPTIONAL**: Uncomment and change embedding model |
| `k` | (commented) `6` | **OPTIONAL**: Uncomment to fix cluster count instead of auto-select |

**Example Modifications**:

```python
# For your own research
from failure_mode_reduction import failure_mode_reduction

result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m18_db.pkl",  # ← Match your model_id
    out_dir="my_summary",  # ← Your output directory
    model_name="all-MiniLM-L6-v2",  # ← Uncomment to specify embedding model
    k=None,  # ← None for auto-select, or set a number (e.g., k=5)
    verbose=True,  # ← See detailed progress
)

print(f"Optimal K: {result['k']}")
print(f"Silhouette scores: {result['silhouette_scores']}")
print(f"Output files: {result['paths']}")
print("\nClustered failure modes:")
print(result["df_clustered"].head(10))
```

**To Run**:
```bash
# After modifying the parameters
python failure_mode_reduction_test.py
```

### Running Tests in Sequence

For a complete test workflow:

```bash
# Step 1: Generate analysis (modify path first!)
python failure_mode_generator_test.py

# Step 2: Cluster results (ensure pickle path matches)
python failure_mode_reduction_test.py
```

### Common Modifications for Research

#### Scenario 1: Different Data Location
```python
# In failure_mode_generator_test.py
res = process_trajectories(
    traj_root_base="/path/to/your/data",  # ← Change this
    model_id=18,
)
```

#### Scenario 2: Different Model
```python
# In failure_mode_generator_test.py
res = process_trajectories(
    traj_root_base="./trajectories",
    model_id=20,  # ← Different model
)

# In failure_mode_reduction_test.py (must match!)
result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m20_db.pkl",  # ← Update m20
    out_dir="summary",
)
```

#### Scenario 3: Fixed Cluster Count
```python
# In failure_mode_reduction_test.py
result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m18_db.pkl",
    out_dir="summary",
    k=7,  # ← Force 7 clusters instead of auto-select
)
```

#### Scenario 4: Different Embedding Model
```python
# In failure_mode_reduction_test.py
result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m18_db.pkl",
    out_dir="summary",
    model_name="paraphrase-MiniLM-L6-v2",  # ← Different embedding model
)
```

### Troubleshooting Test Files

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: [Errno 2] No such file or directory` | Wrong `traj_root_base` path | Update path to your actual trajectory directory |
| `FileNotFoundError: combined_m12_db.pkl` | Pickle file doesn't exist or wrong model ID | Run generator test first, or update model ID in path |
| `ModuleNotFoundError: No module named 'reactxen'` | Missing dependency for LLM calls | Install required package or modify `utils.py` to use your LLM |
| `KeyError: 'addi_fm_cnt'` | Pickle file from different version | Regenerate pickle files with current code |
| Clustering fails with small data | Not enough failure modes to cluster | Ensure you have at least 2 additional failure modes |

## 📊 Performance Considerations

- **LLM Calls**: Generation step makes one LLM call per trajectory (can be slow/expensive)
- **Caching**: Pickle files enable re-running clustering without re-analyzing
- **Batch Processing**: Process trajectories in batches if memory is limited
- **Parallel Processing**: Consider parallelizing LLM calls for large datasets

## 🔄 Pipeline Execution Flow

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Trajectory JSON Files                  │
│  (Located in traj_root_base directory, recursively discovered)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: failure_mode_generator.py                   │
│                                                                   │
│  1. _load_all_json_files() - Recursively load all JSONs         │
│  2. For each trajectory:                                         │
│     a. Format as prompt (Question → Thought/Action/Obs → Answer) │
│     b. get_llm_answer_from_json() - Call LLM API                │
│     c. extract_json_from_response() - Parse JSON response       │
│     d. Extract 14 predefined failure modes (boolean)             │
│     e. Extract additional failure modes (list of dicts)          │
│  3. Store in DataFrame with columns:                             │
│     - model_id, counter, timestamp, vendor, model, ut_id         │
│     - 14 boolean columns for predefined failure modes            │
│     - addi_fm_cnt, addi_fm_list                                  │
│  4. Save per-timestamp pickle: {timestamp}_m{model_id}_db.pkl   │
│  5. Combine all timestamps into: combined_m{model_id}_db.pkl    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           OUTPUT: processed_trajectories/ folder                 │
│  - Individual pickle files per timestamp                         │
│  - Combined pickle file (input for next step)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: failure_mode_reduction.py                   │
│                                                                   │
│  1. Load combined pickle file                                    │
│  2. Filter rows where addi_fm_cnt > 0                           │
│  3. Explode addi_fm_list into separate rows                     │
│  4. Normalize to title/description format                        │
│  5. Save raw CSV: addtional_fm.csv                              │
│  6. Generate embeddings using SentenceTransformer               │
│  7. Determine optimal K:                                         │
│     - If k=None: Test k_min to k_max, select best silhouette    │
│     - If k provided: Use that value                              │
│  8. Perform K-Means clustering                                   │
│  9. For each cluster:                                            │
│     - Find point closest to centroid                             │
│     - Use its title as cluster representative                    │
│  10. Add cluster and failure_mode columns                        │
│  11. Save clustered CSV: additional_fm_clustered.csv            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: summary_codabench/ folder                   │
│  - addtional_fm.csv (raw additional failure modes)              │
│  - additional_fm_clustered.csv (with cluster assignments)       │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Timeline Example

For a dataset with 100 trajectories:

```
Time    Step                                    Status
─────────────────────────────────────────────────────────────
00:00   Start pipeline
00:01   Load 100 JSON files                     ✓ Complete
00:02   Analyze trajectory 1/100 with LLM       Processing...
00:03   Analyze trajectory 2/100 with LLM       Processing...
...     (1-2 seconds per trajectory)
03:20   Analyze trajectory 100/100 with LLM     ✓ Complete
03:21   Save per-timestamp pickles              ✓ Complete
03:22   Save combined pickle                    ✓ Complete
03:23   Load combined pickle                    ✓ Complete
03:24   Explode additional failure modes        ✓ Complete
03:25   Generate embeddings (50 unique FMs)     Processing...
03:26   Test K=2 (silhouette: 0.45)
03:27   Test K=3 (silhouette: 0.52)
03:28   Test K=4 (silhouette: 0.58)
03:29   Test K=5 (silhouette: 0.61) ← Best
03:30   Test K=6 (silhouette: 0.55)
03:31   Test K=7 (silhouette: 0.48)
03:32   Perform final clustering with K=5       ✓ Complete
03:33   Assign representative titles            ✓ Complete
03:34   Save CSVs                               ✓ Complete
03:35   Pipeline complete                       ✓ Success
```

**Total Time**: ~3.5 minutes (mostly LLM calls)

### Key Decision Points

| Decision Point | Logic | Impact |
|----------------|-------|--------|
| **Auto-discover timestamps** | If `timestamps=None`, use all subdirectories | Processes entire dataset vs. specific time periods |
| **Retry on LLM failure** | Max 2 attempts per trajectory | Handles transient API errors |
| **Skip invalid JSON** | Silent skip with try/except | Robust to malformed files |
| **Auto-select K** | If `k=None`, test range and use silhouette | Optimal clustering vs. fixed cluster count |
| **Handle small samples** | Special cases for n=0, n=1, n=2 | Prevents clustering errors |
| **Representative selection** | Closest to centroid | Most typical example per cluster |

## 🤝 Contributing

When adapting this code for your research:

1. **Cite appropriately** if publishing results
2. **Document modifications** to failure mode definitions
3. **Share insights** about new failure patterns discovered
4. **Report issues** with edge cases or unexpected behaviors

## 📄 License

[Add your license information here]

## 📧 Contact

[Add contact information for questions/support]

## 🙏 Acknowledgments

This tool is designed for analyzing multi-agent LLM systems and identifying failure patterns to improve system reliability and performance.

---

**Note**: The `processed_trajectories/` and `summary_codabench/` folders are auto-generated. You can safely delete `summary_codabench/` to regenerate clustering results, but deleting `processed_trajectories/` requires re-running expensive LLM analysis.