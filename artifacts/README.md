# artifacts/

Model weights that do not come from the HuggingFace Hub. Two roots with
opposite rules, deliberately siblings rather than nested, so one can be mounted
read-only while the other stays writable.

```
artifacts/
  tsfm_models/      read-only. Committed here, baked into the image.
  output/
    tuned_models/   writable. For checkpoints an agent fine-tunes during a trial. Never committed.
```

## tsfm_models/ - shipped checkpoints

It holds four TinyTimeMixer checkpoints (`ttm_512_96`, `ttm_512_720`,
`ttm_1536_96`, `ttm_1536_720`) and `ttm_energy_168_24`, a TTM fine-tuned for
energy load forecasting. A fine-tuned checkpoint that ships with the repo
belongs here, not in `output/`, whose contents are never committed or baked
into the image. It can carry a `meta.json` beside the weights for what they
cannot say themselves (domain, lineage, training data), which
`benchmarks/harbor/scripts/generate_model_catalog.py` reads.

Each entry is a `save_pretrained` directory (`config.json` plus weights) that a
catalog card points at:

```json
"source": "local_artifact",
"hf_repo": null,
"artifact_path":    "artifacts/tsfm_models/ttm_512_96",
"model_checkpoint": "artifacts/tsfm_models/ttm_512_96",
"params": { "model_path": "artifacts/tsfm_models/ttm_512_96" }
```

All four fields must name the same location. Only `params.model_path` is read
at load time; the others are metadata that drifts silently if you let it, and
the agent copies the shape it sees here when it registers its own cards.

The path is relative, so it resolves against the process's working directory:
the repo root locally, `/opt/aob` in the container. If the directory is not
there, `from_pretrained` falls back to treating the string as a Hub repo id.
A three-segment path then fails with "Repo id must be in the form ...", and a
two-segment one quietly tries to download from huggingface.co. Neither error
mentions a missing directory, so verify instead of guessing:

```bash
uv run python benchmarks/harbor/scripts/preload_models.py --check
```

That resolves every active card, Hub and local alike, and exits non-zero when
anything would fail at fit time.

### What belongs here

Public, redistributable weights only. Two separate questions, and the second is
easy to miss: the base model's licence, and what the checkpoint was fine-tuned
on. A model tuned on internal or customer data is derived from that data and
does not belong in a public repository whatever the base model's licence says.
Those go in the private set instead.

Plain git is fine at these sizes: the weights here run from 0.15 MB to 20 MB.
Reach for Git LFS if a file approaches 50 MB.

## output/tuned_models/ - agent output

The place for checkpoints an agent fine-tunes during a trial: `run_recipe`'s
`save_to` names the directory, and `register_finetuned` then points a new card
at it. Nothing enforces the location, since `save_to` takes any path. Created
inside the trial container, so it is isolated per trial and discarded with it. `.gitignore` and `.dockerignore` both exclude it: nothing in
here is an input to anything.

Cards for these models carry `created_by: "agent.tsfm.finetune"`, which is what
`--check` uses to skip them. `provenance: "finetuned"` is not the test, because
a shipped checkpoint under `tsfm_models/` can legitimately be fine-tuned too.

If you ever mount this root from the host to make checkpoints outlive a trial,
give each trial its own subdirectory. A shared writable path is how parallel
trials start overwriting each other, which is the same failure the per-trial
CouchDB exists to prevent.
