# artifacts/

Model weights that do not come from the HuggingFace Hub. Two roots with
opposite rules, deliberately siblings rather than nested, so one can be mounted
read-only while the other stays writable.

```
artifacts/
  tsfm_models/      read-only. Committed here, baked into the image.
  output/
    tuned_models/   writable. An agent writes here during a trial. Never committed.
```

## tsfm_models/ - shipped checkpoints

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

Plain git is fine at these sizes. A TTM checkpoint is low single-digit MB.
Reach for Git LFS if a file approaches 50 MB.

## output/tuned_models/ - agent output

`run_recipe`'s `save_to` writes here, and `register_finetuned` then points a new
card at it. Created inside the trial container, so it is isolated per trial and
discarded with it. `.gitignore` and `.dockerignore` both exclude it: nothing in
here is an input to anything.

Cards for these models carry `created_by: "agent.tsfm.finetune"`, which is what
`--check` uses to skip them. `provenance: "finetuned"` is not the test, because
a shipped checkpoint under `tsfm_models/` can legitimately be fine-tuned too.

If you ever mount this root from the host to make checkpoints outlive a trial,
give each trial its own subdirectory. A shared writable path is how parallel
trials start overwriting each other, which is the same failure the per-trial
CouchDB exists to prevent.
