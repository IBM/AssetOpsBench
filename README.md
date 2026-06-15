# Industrial Automation Challenge Starter Kit

[![ProjectPage](https://img.shields.io/badge/Industrial_Automation_Challenge-Page-red)](https://sites.google.com/view/ai-industrial-challenge-ijcai/home) [![Q&A](https://img.shields.io/badge/Question-Answer-blue)](QA.md)

A simple CSV submission framework for the IJCAI 2026 Industrial Automation Challenge. This branch is competition-only and contains the starter-kit files needed to load challenge data, run a participant predictor, and generate a Kaggle-ready submission CSV.

## Updates

2026.06.14: Added the public starter-kit workflow, Q&A page, config files, and CSV submission utilities.

## Quick Start

### Installation Dependencies

```bash
pip install -r requirements.txt
```

### Generate a submission CSV

Update `metadata_config_test.json` with your local dataset path and predictor path, then run:

```bash
python run.py --config metadata_config_test.json
```

The generated submission file is written to:

```text
competition_results/submission.csv
```

## Project Structure

```text
.
├── README.md                 # Starter-kit documentation
├── QA.md                     # Public competition Q&A
├── eval_framework.py         # Submission framework
├── dataset_utils.py          # Dataset loading utilities
├── run.py                    # Command-line submission script
├── metadata_config_val.json  # Example validation config
├── metadata_config_test.json # Example test config
├── requirements.txt          # Python dependencies
└── competition_results/      # Output directory created by the runner
```

## Kaggle Tracks

The challenge has two tracks:

- Track 1
- Track 2

Kaggle links will be added when the final competition pages are available.

Both tracks use the same input data filename on Kaggle. Download the data file from the Kaggle Data tab for the track you are entering, then set `dataset.dataset_path` in the config to the local path of that file.

## Dataset Configuration

Set the dataset path in `metadata_config_val.json` or `metadata_config_test.json`:

```json
{
  "dataset": {
    "dataset_name": "industrial_automation_challenge_test",
    "dataset_path": "path/to/kaggle_dataset_file.jsonl",
    "description": "Industrial Automation Challenge test questions"
  }
}
```

The placeholder path should be replaced with the local path to the downloaded Kaggle data file.

## Usage Examples

### Basic run with config file

```bash
python run.py --config metadata_config_test.json
```

### Override the dataset path

```bash
python run.py \
  --config metadata_config_test.json \
  --dataset-path path/to/kaggle_dataset_file.jsonl
```

### Use your own Python predictor

```bash
python run.py \
  --config metadata_config_test.json \
  --predictor path/to/your_predictor.py:predict
```

The predictor function receives a scenario object with `id`, `text`, and `metadata` fields. It should return an answer letter:

```python
def predict(scenario):
    return {"answer": "A"}
```

### Use an existing command-line agent

```bash
python run.py \
  --config metadata_config_test.json \
  --agent-command 'your-agent --question {question_json}'
```

If the command prints JSON with `answer`, `choice`, or `prediction`, the runner uses that value. Otherwise stdout is used as the answer value.

## Config File

Example config:

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

## Question Type Support

The framework supports multiple-choice industrial reasoning rows with these participant-facing fields:

- `id`: question identifier
- `question_type`: question type string
- `passage`: optional context passage
- `question`: question text
- `options`: answer options keyed by option letter
- `metadata`: optional public metadata

The loader also supports simple rows with `id` and `text`.

## Output Format

The submission CSV contains:

- `id`: question identifier
- `answer`: selected answer letter

## Support

For issues and questions:

1. Check the error message from the runner.
2. Review the config examples in this repository.
3. Open a GitHub issue or use the Kaggle discussion page.

Happy competing!
