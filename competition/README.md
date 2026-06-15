# Industrial Automation Challenge Starter Kit

[![ProjectPage](https://img.shields.io/badge/Industrial_Automation_Challenge-Page-red)](https://sites.google.com/view/ai-industrial-challenge-ijcai/home) [![Kaggle Track 1](https://img.shields.io/badge/Kaggle-Track_1-green)](https://www.kaggle.com/competitions/industrial-automation-challenge-track-1) [![Kaggle Track 2](https://img.shields.io/badge/Kaggle-Track_2-green)](https://www.kaggle.com/competitions/industrial-automation-challenge-track-2) [![Q&A](https://img.shields.io/badge/Question-Answer-blue)](QA.md)

A simple inference framework for the IJCAI 2026 Industrial Automation Challenge. This starter kit provides an easy-to-use interface for generating submission data in CSV format and packaging the metadata file used by the competition workflow.

## Updates

2026.06.14: Added the public starter-kit workflow, Q&A page, metadata configs, and submission packaging utilities.

## Quick Start

### Installation Dependencies

From the repository root:

```bash
pip install -r competition/requirements.txt
```

### Generate a submission package

Update `metadata_config_test.json` with your dataset path and predictor path, then run:

```bash
python competition/run.py --config competition/metadata_config_test.json
```

## Project Structure

```text
competition/
├── README.md                 # Starter-kit documentation
├── QA.md                     # Public competition Q&A
├── eval_framework.py         # Main submission framework
├── dataset_utils.py          # Dataset loading utilities
├── run.py                    # Command-line submission script
├── metadata_config_val.json  # Example validation config
├── metadata_config_test.json # Example test config
├── requirements.txt          # Python dependencies
└── competition_results/      # Output directory created by the runner
```

## Dataset Preparation

Download the challenge datasets from Kaggle:

```text
https://www.kaggle.com/competitions/industrial-automation-challenge-track-1
https://www.kaggle.com/competitions/industrial-automation-challenge-track-2
```

Configure the dataset path in `metadata_config_val.json` or `metadata_config_test.json`:

```json
{
  "dataset": {
    "dataset_name": "industrial_automation_challenge_test",
    "dataset_path": "path/to/challenge_dataset.jsonl",
    "description": "Industrial Automation Challenge questions"
  }
}
```

## Usage Examples

### Basic run with config file

After updating the config paths:

```bash
python competition/run.py --config competition/metadata_config_test.json
```

### Override the dataset path

```bash
python competition/run.py   --config competition/metadata_config_test.json   --dataset-path path/to/challenge_dataset.jsonl
```

### Use your own Python predictor

```bash
python competition/run.py   --config competition/metadata_config_test.json   --predictor path/to/your_predictor.py:predict
```

The predictor function receives a scenario object with `id`, `text`, and `metadata` fields. It should return a dictionary containing an answer letter:

```python
def predict(scenario):
    return {
        "answer": "A",
        "prediction": "A",
        "reasoning": "Optional rationale",
        "trajectory": []
    }
```

### Use an existing command-line agent

```bash
python competition/run.py   --config competition/metadata_config_test.json   --agent-command 'your-agent --question {question_json}'
```

If the command prints JSON with `answer`, `choice`, or `prediction`, the runner uses that value. Otherwise stdout is used as the prediction.

## Configuration

Create a metadata config file. Example:

```json
{
  "metadata": {
    "model_name": "my-industrial-model",
    "model_type": "CustomModel",
    "track": "internal_reasoning",
    "base_model_type": "OpenWeighted",
    "base_model_name": "my-base-model",
    "dataset": "industrial_automation_challenge_test",
    "additional_info": "Submission using configuration file",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  },
  "dataset": {
    "dataset_name": "industrial_automation_challenge_test",
    "dataset_path": "path/to/challenge_dataset.jsonl",
    "description": "Industrial Automation Challenge questions"
  },
  "output_dir": "competition_results",
  "output_file": "submission.csv"
}
```

### Required Metadata Fields

- `model_name`: Display name of your model
- `track`: Either `internal_reasoning` or `agentic_reasoning`
- `base_model_type`: `API`, `OpenWeighted`, or `Hybrid`
- `base_model_name`: Name of the underlying model
- `dataset`: Name of the dataset

The following fields can be left empty for early runs and completed for final submissions when applicable: `additional_info`, `average_tokens_per_question`, `average_tools_per_question`, and `tool_category_coverage`.

## Question Type Support

The framework supports multiple-choice industrial reasoning rows with these fields:

```json
{
  "id": "example-001",
  "question_type": "open_ended_multi_choice",
  "passage": "Optional context passage",
  "question": "Question text",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "metadata": {}
}
```

The loader also supports simple rows with `id` and `text`.

## Output Format

The framework generates a CSV file and a zip package containing metadata. The default Kaggle CSV structure is:

- `id`: Question identifier
- `answer`: Selected answer letter

The zip package contains:

```text
submission.csv
meta_data.json
```

Example metadata package:

```json
{
  "meta_data": {
    "model_name": "my-industrial-model",
    "track": "internal_reasoning",
    "model_type": "CustomModel",
    "base_model_type": "OpenWeighted",
    "base_model_name": "my-base-model",
    "dataset": "industrial_automation_challenge_test",
    "additional_info": "Submission using configuration file",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  }
}
```

## Support

For issues and questions:

1. Check the error message from the runner.
2. Review the config examples in this folder.
3. Open a GitHub issue or use the Kaggle discussion page.

Happy competing!
