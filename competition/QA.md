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

## Should submissions include token usage?

Not in the current public submission schema. Submit only the columns announced in
Kaggle and in `competition/examples/sample_submission.csv`.

If token-efficiency becomes part of live scoring, the organizers will announce a
separate participant-facing field in the rules and sample submission. Do not add
extra private/evaluation metadata columns to your Kaggle submission unless the
rules explicitly request them.

## Are answers allowed in public data?

Final test data should not include answer labels. Organizers may provide a
labeled validation artifact if desired, but participant-facing test data should
contain prompt information only.

## Which fields are private and should not be in public test data?

Do not publish private/evaluation fields in participant test data, including:

- answer, label, or reference-answer fields
- rubric, characteristic-form, or scoring-method fields
- `usage` or any other reserved evaluation metadata

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
