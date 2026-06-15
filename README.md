# Industrial Automation Challenge: Benchmarking Physics-Grounded LLMs for Task Reasoning

[![IJCAI 2026](https://img.shields.io/badge/IJCAI_2026-Competition-red)](https://2026.ijcai.org/competitions/#:~:text=Industrial%20Automation%20Challenge%3A%20Benchmarking%20Physics%2DGrounded%20LLMs%20for%20Task%20Reasoning) [![Project Page](https://img.shields.io/badge/Industrial_Automation_Challenge-Page-red)](https://sites.google.com/view/ai-industrial-challenge-ijcai/home) [![Q&A](https://img.shields.io/badge/Question-Answer-blue)](QA.md)

This repository contains the public starter kit for the IJCAI 2026 Industrial Automation Challenge: Benchmarking Physics-Grounded LLMs for Task Reasoning.

Participants can use this starter kit to load the Kaggle data file, connect their own predictor, and generate a Kaggle-ready CSV submission.

## Updates

2026.06.14: Added the public starter-kit workflow, Q&A page, config files, and CSV submission utilities.

## Repository Structure

```text
.
├── README.md                 # Competition starter-kit documentation
├── QA.md                     # Public competition Q&A
├── dataset_utils.py          # Dataset loading utilities
├── eval_framework.py         # Submission-generation framework
├── run.py                    # Command-line entry point
├── metadata_config_val.json  # Validation config template
├── metadata_config_test.json # Test config template
└── requirements.txt          # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Tracks

The challenge has two Kaggle tracks:

- Track 1
- Track 2

Both tracks use the same Kaggle input data filename. Download the data file from the Kaggle Data tab for the track you are entering, then update the config with your local file path.

The final Kaggle links can be added to this README once they are available.

## Dataset Config

Update `metadata_config_val.json` or `metadata_config_test.json`:

```json
{
  "dataset": {
    "dataset_name": "industrial_automation_challenge_test",
    "dataset_path": "path/to/kaggle_dataset_file.jsonl",
    "description": "Industrial Automation Challenge test questions"
  },
  "predictor": {
    "path": "path/to/your_predictor.py:predict"
  },
  "submission_columns": ["id", "answer"],
  "output_dir": "competition_results",
  "output_file": "submission.csv"
}
```

Replace `path/to/kaggle_dataset_file.jsonl` with the local path to the Kaggle data file.

## Input Rows

The loader supports multiple-choice rows with participant-facing fields such as:

- `id`: question identifier
- `question_type`: question type string
- `passage`: context passage, when present
- `question`: question text
- `options`: answer options keyed by option letter
- `metadata`: public row metadata, when present

Rows with only `id` and `text` are also supported.

## Predictor Interface

Your predictor should be a Python function that accepts one scenario and returns an answer. Example:

```python
def predict(scenario):
    return {"answer": "A"}
```

A scenario object has:

- `scenario.id`
- `scenario.text`
- `scenario.options`
- `scenario.metadata`
- `scenario.to_dict()`

You can pass the predictor as `path/to/file.py:function_name`.

## Generate a Submission

```bash
python run.py --config metadata_config_test.json
```

The runner writes:

```text
competition_results/submission.csv
```

The CSV format is:

```csv
id,answer
```

## Useful Commands

Run with the validation config:

```bash
python run.py --config metadata_config_val.json
```

Override the dataset path:

```bash
python run.py \
  --config metadata_config_test.json \
  --dataset-path path/to/kaggle_dataset_file.jsonl
```

Override the predictor path:

```bash
python run.py \
  --config metadata_config_test.json \
  --predictor path/to/your_predictor.py:predict
```

Use a command-line agent:

```bash
python run.py \
  --config metadata_config_test.json \
  --agent-command 'your-agent --question {question_json}'
```

If the command prints JSON with `answer`, `choice`, or `prediction`, the runner uses that value. Otherwise, stdout is used as the answer.

## Submission Format

Submit a CSV file with exactly these columns:

- `id`
- `answer`

`answer` should be the selected option letter for that row.

## Support

For questions, use the competition Q&A/discussion channel or open a GitHub issue.
