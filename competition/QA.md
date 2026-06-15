# Industrial Automation Challenge Q&A

This page collects public clarifications for the AssetOpsBench / Industrial
Automation Challenge competition branch.

## Where are the competition pages?

- Website: https://sites.google.com/view/ai-industrial-challenge-ijcai/home
- Kaggle Track 1: https://www.kaggle.com/competitions/industrial-automation-challenge-track-1
- Kaggle Track 2: https://www.kaggle.com/competitions/industrial-automation-challenge-track-2

## What is Track 1?

Track 1 is internal model reasoning. Models must answer using only their
internal parameters. External tools, internet access, retrieval systems, and
structured databases are not allowed during inference.

## What is Track 2?

Track 2 is agentic tool-augmented reasoning. Models may operate as agents and
invoke industrial tools such as asset documentation, sensor telemetry, and
maintenance/work-order logs. Participants should preserve a reasoning/tool-call
trace for organizer-side review where requested.

## What should the Kaggle submission look like?

The Kaggle-friendly MCQA submission schema is:

```csv
id,answer
q123,A
```

`answer` should be the selected option letter.

## Why is there a `usage` column in the solution file?

In Kaggle solution files, `usage` is the leaderboard split marker. It indicates
whether each hidden row contributes to the Public or Private leaderboard. It is
not token usage, cost, latency, or tool usage.

Participants should not include `usage` in public datasets or submissions. If
the competition later needs live token-efficiency scoring, use a separate field
name such as `token_usage` or `total_tokens`.

## Are answers allowed in public data?

Final test data should not include answer labels. Organizers may provide a
labeled validation set if desired, but the public test set and hidden solution
must remain separated.

## Which fields are private and should not be in public test data?

Do not publish fields such as `answer`, `correct_answer`, `expected_answer`,
`ground_truth`, `reference_answer`, `rubric`, `characteristic_form`,
`scoring_method`, or `usage` in participant test data.

## How are submissions scored?

The current starter-kit metric scores exact MCQA answer accuracy from `id` and
`answer`. The IJCAI proposal also mentions latency, reasoning completeness, and
token efficiency. Those broader criteria require auditable logs or separate
fields and are best handled during final/offline review unless the Kaggle metric
schema is expanded.

## What model constraints apply?

The IJCAI proposal states that participants are limited to open-source base
models with no more than 8B parameters. Track-specific rules may add further
constraints.

## What files should participants use from this repo?

- `competition/readme.md` for the workflow
- `competition/metadata_config_phase1.json` for Track 1 examples
- `competition/metadata_config_phase2.json` for Track 2 examples
- `competition/examples/baseline_predictor.py` as a minimal predictor template
- `competition/examples/sample_submission.csv` as the toy submission format
