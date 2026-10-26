# DiagnosticIQ dataset creation

The `dataset_utils` package and the accompanying notebooks build the
DiagnosticIQ multiple-choice benchmark and its variants from expert-authored
fault-detection rule documents. The generation algorithms were transferred from
the original DiagnosticIQ code base without behavioral changes.

## Source data is not part of this repository

This repository contains only the code. The rule documents, every artifact
derived from them, and the generated datasets live in an external workspace
named by `ASSETOPSBENCH_DATASET_DIR` (see [Setup](#setup)):

- The notebooks are committed **without outputs**. Once run, their outputs
  contain rule conditions, observations and questions, so clear them before
  committing, e.g.
  `jupyter nbconvert --clear-output --inplace notebook/<name>.ipynb`.
- Debug listings and figures are written to the workspace, never to the
  repository.

## Variants

Names follow the paper. The DiagnosticIQ notebooks are in `notebook/` and
their names start with `diagnosticiq_`. Each notebook reads from and writes to
`$ASSETOPSBENCH_DATASET_DIR/datasets/` unless noted.

| Paper name | Notebook | Reads | Writes | V3.1 release folder |
| --- | --- | --- | --- | --- |
| DiagnosticIQ | `diagnosticiq_simple_dataset_creation.ipynb` | `extracted/` artifacts | `DiagnosticIQ` | `DiagnosticIQV3.1` |
| DiagnosticIQ Pro | `diagnosticiq_complex_dataset_creation.ipynb` | `DiagnosticIQ` | `DiagnosticIQPro` | `DiagnosticIQProV3.1` |
| DiagnosticIQ Pert | `diagnosticiq_simple_perteval_simple_dataset_creation.ipynb` | `DiagnosticIQPro` | `DiagnosticIQPert` | `DiagnosticIQPertV3.1` |
| DiagnosticIQ Verbose | `diagnosticiq_verbose_dataset_creation.ipynb` | `DiagnosticIQ`, `verbose/` | `DiagnosticIQVerbose` | `DiagnosticIQVerbV3.1` |
| Test rules | `diagnosticiq_simple_test_dataset_creation.ipynb` | `test_datasets/`, `DiagnosticIQ` | `test_datasets/datasets/DiagnosticIQTest` | — |

DiagnosticIQ Aug and DiagnosticIQ Rationale are not produced by this code.

## Pipeline

### 1. Source artifacts

These notebooks turn the source documents into the inputs of question
generation. Run them in order; each one's original name in the DiagnosticIQ
code base is given in parentheses.

| Step | Notebook | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `diagnosticiq_source_rules_extraction.ipynb` (`scrap_pdf`) | `data/skyspark_rules.pdf` | `extracted/extracted_data.json` |
| 2 | `diagnosticiq_source_rules_consistency_check.ipynb` (`consistency_check`) | `extracted/extracted_data.json` | `extracted/observations.rules.json` and three files of incomplete rules |
| 3 | `diagnosticiq_source_rules_formatting.ipynb` (`data_formatting`) | the step 2 files | the same files, in place: adds `asset`, splits observations into lists |
| 4 | `diagnosticiq_source_rules_formalizing.ipynb` (`formalizing_rules`) | `extracted/observations.rules.json` | `extracted/TreeStruct.pkl` (rule condition trees) |
| 5 | `diagnosticiq_source_asset_descriptions.ipynb` (`asset_stats`) | `extracted/observations.rules.json`, `data/alert_events.csv`, `data/equipment_profile.csv`, `data/GRE_Asset_Description.csv` | `extracted/equipment_info/gre_desc.json`, `rule_to_equipment_map.json` |
| test | `diagnosticiq_source_test_rules_conversion.ipynb` (`test_datasets/conversions`) | `test_datasets/rule_logic_test2.json` | `test_datasets/TEST_rule_information.json`, `test_datasets/TEST_TreeStruct.pkl` |

Step 1 was a legacy notebook marked *DO NOT RUN* in the source code base,
because running it again overwrites `extracted_data.json`. Step 3 rewrites its
inputs in place, so run it exactly once on fresh step 2 output.

### 2. Question generation

`diagnosticiq_simple_dataset_creation.ipynb` builds DiagnosticIQ:

1. **Unique observations.** Observation texts are embedded with
   `all-MiniLM-L6-v2`; pairs above the 0.992 similarity quantile are merged into
   one canonical observation (the longest in the group).
2. **Rule-to-rule similarity (RRSim).** Each rule's condition text is embedded
   with `all-mpnet-base-v2`; rules are ranked by cosine similarity.
3. **Question creation.** `creation.create_dataset` expands each rule's
   condition tree into conjunctive trigger scenarios and, for each one, writes
   *selection* questions (the correct option is one of the rule's own
   observations; distractors come from the least similar rules) and
   *elimination* questions (the correct option comes from the least similar
   rules; the rest are the rule's own observations). Prompts are drawn from
   the template pools in `creation.py`.
4. **Question formatting.** `outputs.question_formatting` writes the final
   question text (asset description, conditions, how long they were met, and
   the prompt) and sets `question_first` and `text_type`.

`DatasetCreationConfig` controls step 3:

| Paper symbol | Config key | Meaning | Default |
| --- | --- | --- | --- |
| — | `num_options` | options per question | 3 |
| α | `sel_random_sample` | distractor samples per rule for selection | 10 |
| β | `elem_random_sample` | correct-option samples per rule for elimination | 25 |
| N_sel_topk, N_eli_topk | `n_least_sim_rules` | least similar rules to draw options from | 25 |
| — | `question_template_mode`, `question_template_random_count` | how prompts are picked from the template pools | `pre-select`, 1 |

The variants are then derived from DiagnosticIQ:

- **Pro** expands every question to up to 10 options with observations from
  the 35 least similar rules (selection) or from the rule's own observations
  (elimination).
- **Pert** applies four perturbations to each question: shifted option labels
  (Caesar shift of 15), parenthesized labels, a different prompt from the same
  template pool, and shuffled conditions and options.
- **Verbose** paraphrases each question's conditions. Ten representative
  questions are picked by clustering condition embeddings (`all-mpnet-base-v2`,
  0.988 per-row similarity quantile); their conditions are paraphrased by hand
  into `verbose/condition_verberlizer_examples.txt`, and those examples
  prompt `mistralai/mistral-large` on watsonx.ai to paraphrase the rest.
- **Test rules** run the DiagnosticIQ pipeline on a small hand-written rule set.

Question formatting is applied wherever a notebook writes or changes the
fields the question text is built from: DiagnosticIQ and Test rules after
creation, Pert after perturbing, and Verbose with the paraphrased conditions
in place of the symbolic list. Pro keeps those fields, so it keeps the
formatted text it inherits.

> [!IMPORTANT]
> **Configuration of the released DiagnosticIQ V3.1 (used in the paper)**
>
> The notebooks run the generic configuration above. The V3.1 release used:
>
> | Setting | V3.1 value |
> | --- | --- |
> | `num_options` | 4 |
> | `sel_random_sample` (α) | 10 |
> | `elem_random_sample` (β) | **10** (the default is 25) |
> | `n_least_sim_rules` | 25 |
> | `question_template_mode` / `question_template_random_count` | `random` / 1 |
> | `random.seed` / `np.random.seed` | 420 |
> | Pro: options / least similar rules | 10 / 35 |
> | Pert: source dataset | **DiagnosticIQ** (set `source_dataset = "DiagnosticIQ"`; the notebook default is `DiagnosticIQPro`) |
> | Verbose: model and decoding | `mistralai/mistral-large`, temperature 0, `max_new_tokens` 2000, `min_new_tokens` 1, `random_seed` 20, stop sequences `<>` and `Note:` |
>
> To generate DiagnosticIQ with this configuration, add
> `"elem_random_sample": 10` to the `DatasetCreationConfig` in
> `diagnosticiq_simple_dataset_creation.ipynb`. With the V3.1 rule set this gives the
> released composition of 6,690 questions (5,180 selection and 1,510
> elimination).
>
> The V3.1 asset descriptions also cover HXU, PIU, Plate & Frame and VAV,
> which `GRE_Asset_Description.csv` does not; add them to
> `extracted/equipment_info/gre_desc.json` before generating to reproduce the
> released prompts.

## Setup

Install AssetOpsBench and the notebook dependencies from the repository root:

```bash
uv sync --group dataset-creation
```

Point `ASSETOPSBENCH_DATASET_DIR` at the external workspace:

```bash
export ASSETOPSBENCH_DATASET_DIR=/path/to/diagnosticiq-data
```

Put the source documents under `data/` and, for the test rules,
`test_datasets/rule_logic_test2.json`. The notebooks create everything else:

```text
$ASSETOPSBENCH_DATASET_DIR/
├── data/                                  source documents (not distributed)
│   ├── skyspark_rules.pdf
│   ├── alert_events.csv
│   ├── equipment_profile.csv
│   └── GRE_Asset_Description.csv
├── extracted/                             source artifacts (steps 1-5)
│   ├── extracted_data.json
│   ├── observations.rules.json
│   ├── TreeStruct.pkl
│   └── equipment_info/
│       └── gre_desc.json
├── datasets/                              generated datasets
├── test_datasets/
│   ├── rule_logic_test2.json
│   ├── TEST_rule_information.json
│   ├── TEST_TreeStruct.pkl
│   └── datasets/
├── verbose/
│   ├── incontext_expl.json                representative question ids
│   └── condition_verberlizer_examples.txt hand-written paraphrases
└── debug/                                 debug listings
```

The Verbose notebook calls watsonx.ai and needs `WATSONX_APIKEY`, `WATSONX_URL`
and `WATSONX_PROJECT_ID` in the environment or the repository `.env` (see
`.env.public`).

The V3.1 source artifacts were generated with sentence-transformers 4.1.0,
torch 2.7.0 and transformers 4.52.4. This repository locks sentence-transformers
4.1.0, but torch and transformers follow the repository lock. The embedding-based
steps (observation merging and rule-to-rule similarity) can therefore shift
slightly when they are recomputed.

Start Jupyter from the repository root. The notebooks do not depend on the
process working directory:

```bash
uv run --group dataset-creation jupyter lab notebook/
```

## Programmatic use

The package is installed with AssetOpsBench and can be imported from any
working directory:

```python
from dataset_utils import config, creation
from dataset_utils.outputs import question_formatting
from dataset_utils.paths import get_dataset_dir
from dataset_utils.reader import ADIQDataset

datasets_dir = get_dataset_dir() / "datasets"
dataset_config = config.DatasetCreationConfig(
    {
        "num_options": 4,
        "question_template_mode": config.QuestionTemplateMode.random,
        "question_template_random_count": 1,
    }
)

dataset = creation.create_dataset(
    rule_information,
    observation_mapping,
    asset_descriptions,
    rule_id_similarity_map,
    dataset_config,
)
creation.save_dataset(dataset, "DiagnosticIQ", datasets_dir)

ds = ADIQDataset(datasets_dir / "DiagnosticIQ")
for q in ds.questions:
    question_formatting(q, ds.asset_descriptions)
ds.save("DiagnosticIQ", datasets_dir)
```

## Tests

```bash
uv run pytest src/dataset_utils/tests
```

## Existing pickle compatibility

`TreeStruct.pkl` written by `diagnosticiq_source_rules_formalizing.ipynb` refers to
`__main__.Node`, because that notebook defines `Node` inline. Older pickles
refer to `utils.tree.Node` or `dataset.utils.tree.Node`.
`dataset_utils.utils.file_handle.load_pickle` maps all three to
`dataset_utils.utils.tree.Node` while loading.

As with any Python pickle, only load files from a trusted source.
