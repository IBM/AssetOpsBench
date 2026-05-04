# CODS 2025 AssetOpsBench Scenarios

This directory contains the 22 scenarios used in the
CODS 2025 AssetOpsBench-Live competition, organized by
phase. Each phase contains 11 scenarios across five
functional categories:

- **IoT** (2 scenarios): sensor and metadata retrieval
- **FMSR** (2 scenarios): failure mode and sensor relations
- **TSFM** (2 scenarios): time-series forecasting capability
- **WO** (3 scenarios): work-order querying and bundling
- **E2E** (2 scenarios): end-to-end multi-agent reasoning

## Files

- `phase1_development/scenarios.jsonl` — 11 development
  utterances (Phase 1, open competition window).
- `phase2_evaluation/scenarios.jsonl` — 11 evaluation
  utterances (Phase 2, hidden assessment phase).

## Schema (one JSON object per line)

| Field | Type | Description |
|---|---|---|
| `id` | string | Question ID from the AssetOpsBench question bank |
| `category` | string | Functional category (IoT, FMSR, TSFM, WO, E2E) |
| `utterance` | string | The natural-language scenario prompt |
| `phase` | string | `development` or `evaluation` |
| `paired_with` | string | (Phase 2 only) the development question this evaluation paraphrases |
| `source` | string | Canonical Hugging Face dataset reference |

## Pairing Design

Phase 1 and Phase 2 scenarios are paired one-to-one to test
generalization to new phrasings of the same task. The mean
inter-phase semantic similarity (Sentence-BERT cosine) across
all 11 pairs is 0.83, with deliberately wide variation
(0.55 to 0.98) to test agent robustness across paraphrase
distance.

| Dev Q | Eval Q | Category | Similarity |
|---|---|---|---|
| Q5 | Q7 | IoT | 0.797 |
| Q8 | Q11 | IoT | 0.868 |
| Q114 | Q107 | FMSR | 0.914 |
| Q106 | Q108 | FMSR | 0.959 |
| Q203 | Q201 | TSFM | 0.617 |
| Q204 | Q205 | TSFM | 0.555 |
| Q400 | Q403 | WO | 0.855 |
| Q405 | Q410 | WO | 0.984 |
| Q424 | Q411 | WO | 0.764 |
| Q604 | Q605 | E2E | 0.943 |
| Q607 | Q606 | E2E | 0.880 |

## Full Records

These JSONL files contain only the scenario prompts. For
ground-truth answers, scoring rubrics, and any multimodal
inputs (sensor data, work-order tables), refer to the
canonical dataset:

https://huggingface.co/datasets/ibm-research/AssetOpsBench

## Citation

[your BibTeX block]