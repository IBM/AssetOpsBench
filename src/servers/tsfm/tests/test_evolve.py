"""AlphaEvolve loop: ask → (simulated agent mutates) → tell → archive grows + elites improve.

The 'agent' here is a deterministic mutator (no LLM): the server-side loop is what we verify —
validation, evaluation→fitness, MAP-Elites cells, islands, lineage, and elite selection."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from ..core.store import MemoryStore
from ..engine import evolve as E
from ..io import refs

NAIVE = "sktime.forecasting.naive.NaiveForecaster"


def _data():
    return refs.materialize_iot(np.sin(np.arange(160) / 4.0) + 0.01 * np.arange(160), asset_id="evo")


def _single(strategy):
    return {"estimator": {"sktime_class": NAIVE, "params": {"strategy": strategy}}, "fh": [1, 2, 3]}


def _ensemble():
    return {"ensemble": {"members": [{"sktime_class": NAIVE, "params": {"strategy": "last"}},
                                     {"sktime_class": NAIVE, "params": {"strategy": "mean"}}],
                         "combine": "mean"}, "fh": [1, 2, 3]}


def _with_transform():
    return {"estimator": {"sktime_class": NAIVE, "params": {"strategy": "drift"}},
            "transforms": [{"sktime_class": "sktime.transformations.series.detrend.Detrender"}],
            "fh": [1, 2, 3]}


def _tell(store, recipe, ref, parent=None):
    return E.evolve_tell(store, "tsfm_forecasting", "recipe", recipe, parent_id=parent,
                         data_ref=ref, timestamp_column="timestamp", target_columns=["value"])


def test_seed_then_grow_archive_and_cells():
    s, ref = MemoryStore(), _data()
    r0 = _tell(s, _single("last"), ref)
    assert r0["accepted"] and r0["is_new_elite"] and r0["generation"] == 0
    # different STRUCTURE → different behaviour cells (MAP-Elites grid fills)
    r_ens = _tell(s, _ensemble(), ref)
    r_tr = _tell(s, _with_transform(), ref)
    assert r_ens["cell"] != r0["cell"] and r_tr["cell"] != r0["cell"]
    best = E.evolve_best(s, "tsfm_forecasting")
    assert best["n_cells"] >= 3


def test_elite_replacement_within_a_cell():
    s, ref = MemoryStore(), _data()
    a = _tell(s, _single("mean"), ref)
    b = _tell(s, _single("drift"), ref, parent=a["evolve_id"])   # same cell, child
    assert b["cell"] == a["cell"] and b["generation"] == 1
    # exactly one of them is the elite for that cell; archive keeps both for lineage
    elites = E._elites(s, "tsfm_forecasting")
    cell_elite = [e for e in elites if e["cell"] == a["cell"]]
    assert len(cell_elite) == 1 and cell_elite[0]["fitness"] == max(a["fitness"], b["fitness"])


def test_ask_returns_parents_inspirations_and_evidence():
    s, ref = MemoryStore(), _data()
    for rec in (_single("last"), _ensemble(), _with_transform()):
        _tell(s, rec, ref)
    ask = E.evolve_ask(s, "tsfm_forecasting", data_ref=ref, timestamp_column="timestamp",
                       channels=["value"])
    assert ask["parents"] and "program" in ask["parents"][0]
    assert ask["evidence"]["n_observations"] == 160 and ask["contract"]["task_id"] == "tsfm_forecasting"


def test_ask_on_empty_archive_gives_seed_instructions():
    ask = E.evolve_ask(MemoryStore(), "tsfm_forecasting")
    assert ask["parents"] == [] and "empty" in ask["instructions"].lower()


def test_invalid_candidate_rejected_not_archived():
    s, ref = MemoryStore(), _data()
    bad = E.evolve_tell(s, "tsfm_forecasting", "recipe", {"foo": 1}, data_ref=ref,
                        timestamp_column="timestamp", target_columns=["value"])
    # no estimator/ensemble → run_recipe raises → caught as invalid, nothing archived
    assert not bad["accepted"] and not s.find(E.ARCHIVE, {})


def test_feature_program_evolves_and_gates():
    s = MemoryStore()
    code = ("import numpy as np\n"
            "class Transformation:\n"
            "    def fit(self, X, metadata): return {'m': float(np.mean(X))}\n"
            "    def transform(self, X, state): return np.asarray(X, float) - state['m']\n"
            "    def inverse_transform(self, Y, state): return np.asarray(Y, float) + state['m']\n")
    feat = {"feature_id": "evo_center", "class_name": "Transformation", "code": code,
            "invertible": True, "output_type": "centered_series"}
    r = E.evolve_tell(s, "tsfm_forecasting", "feature", feat)
    assert r["accepted"] and r["fitness"] >= 1.0          # passes gate; invertible bonus
    assert E.evolve_best(s, "tsfm_forecasting", kind="feature")["elites"]
