"""A test double implementing the HuggingFace-backed forecaster contract, with no torch.

This exists so the fine-tune -> save -> register -> serve lifecycle can be tested in CI. It is NOT
a stand-in for TTM's behaviour; it reproduces exactly the four things the tsfm server relies on:

  1. a `model_path` constructor param, pointing at a local dir or a repo id
  2. a `fit_strategy` param defaulting to "minimal" - i.e. it TRAINS unless told otherwise,
     which is the sktime TTM default that the regime logic has to cope with
  3. `self.model` after fit, exposing `save_pretrained(dir)`  (what recipe['save_to'] calls)
  4. loading those saved weights back via `from_pretrained(dir)` at the next fit

Its "weights" are a seasonal profile in a JSON file. The arithmetic is irrelevant; the contract is
the point.
"""
import json
import os

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster

CALLS = {"fit": 0, "trained": 0, "loaded_from": []}


def reset():
    CALLS.update({"fit": 0, "trained": 0, "loaded_from": []})


class _FakeHFModel:
    """Stands in for a transformers PreTrainedModel."""

    def __init__(self, profile):
        self.profile = np.asarray(profile, dtype=float)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"model_type": "fake_hf", "sp": len(self.profile)}, f)
        with open(os.path.join(path, "weights.json"), "w") as f:
            json.dump({"profile": self.profile.tolist()}, f)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "weights.json")) as f:
            return cls(json.load(f)["profile"])


class FakeTTMForecaster(BaseForecaster):
    """Named so resolver._FM_KEYS ('ttm') sees it as a foundation model."""

    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.Series",
        "capability:insample": False,
        "X-y-must-have-same-index": False,
    }

    def __init__(self, model_path=None, fit_strategy="minimal", sp=24, training_args=None):
        self.model_path = model_path
        self.fit_strategy = fit_strategy      # default "minimal" == TTM's: it TRAINS
        self.sp = sp
        self.training_args = training_args
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        CALLS["fit"] += 1
        yv = np.asarray(y, dtype=float)

        pretrained = None
        if self.model_path and os.path.isdir(str(self.model_path)):
            pretrained = _FakeHFModel.from_pretrained(self.model_path)   # local checkpoint
            CALLS["loaded_from"].append("checkpoint")
        else:
            pretrained = _FakeHFModel(np.zeros(self.sp))                 # "hub" weights
            CALLS["loaded_from"].append("hub")

        if self.fit_strategy == "zero-shot":
            self.model = pretrained            # use as-is; NO training
        else:
            CALLS["trained"] += 1              # "minimal"/"full" -> adapt to this series
            n = (len(yv) // self.sp) * self.sp
            prof = yv[:n].reshape(-1, self.sp).mean(axis=0) if n else np.zeros(self.sp)
            if self.fit_strategy == "minimal" and pretrained.profile.any():
                prof = 0.5 * prof + 0.5 * pretrained.profile
            self.model = _FakeHFModel(prof)
        self._cut = len(yv)
        return self

    def _predict(self, fh, X=None):
        idx = fh.to_absolute(self.cutoff)
        h = len(idx)
        prof = self.model.profile
        out = [prof[(self._cut + i) % self.sp] for i in range(h)]
        return pd.Series(out, index=idx.to_pandas(), name="y")
