# artifacts/

Model weights that do not come from the HuggingFace Hub.

```
artifacts/
  tsfm_models/      read-only. Committed here, baked into the image.
  output/
    tuned_models/   writable. For checkpoints an agent fine-tunes during a trial. Never committed.
```

## tsfm_models/ - shipped checkpoints

Four TinyTimeMixer checkpoints (`ttm_512_96`, `ttm_512_720`, `ttm_1536_96`,
`ttm_1536_720`) and `ttm_energy_168_24`, a TTM fine-tuned for energy load
forecasting. A fine-tuned checkpoint that ships with the repo belongs here, not
in `output/`. An optional `meta.json` beside the weights records what they
cannot (domain, lineage, training data) for
`benchmarks/harbor/scripts/generate_model_catalog.py`.

Each entry is a `save_pretrained` directory (`config.json` plus weights) that a
catalog card points at:

```json
"source": "local_artifact",
"hf_repo": null,
"artifact_path":    "artifacts/tsfm_models/ttm_512_96",
"model_checkpoint": "artifacts/tsfm_models/ttm_512_96",
"params": { "model_path": "artifacts/tsfm_models/ttm_512_96" }
```

Keep the location fields in step. Only `params.model_path` is read at load
time, but the agent copies this shape when it registers its own cards.

The path resolves against the working directory: the repo root locally,
`/opt/aob` in the container. A missing directory is treated as a Hub repo id,
and the resulting error does not mention the directory, so check instead:

```bash
uv run python benchmarks/harbor/scripts/preload_models.py --check
```

That resolves every active card, Hub and local alike, and exits non-zero when
anything would fail at fit time.

### What belongs here

Public, redistributable weights only: check both the base model's licence and
what the checkpoint was fine-tuned on. A model tuned on internal or customer
data belongs in the private set, whatever the base licence says.

Plain git is fine at these sizes (0.15 MB to 20 MB); use Git LFS if a file
approaches 50 MB.

## output/tuned_models/ - agent output

For checkpoints an agent fine-tunes during a trial: `run_recipe`'s `save_to`
names the directory (any path works), and `register_finetuned` points a new
card at it. It lives in the trial container and is discarded with it;
`.gitignore` and `.dockerignore` both exclude it.

Cards for these models carry `created_by: "agent.tsfm.finetune"`, which is what
`preload_models.py --check` uses to skip them.

If you mount this root from the host, give each trial its own subdirectory, or
parallel trials will overwrite each other.
