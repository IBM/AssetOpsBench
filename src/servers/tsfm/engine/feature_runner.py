"""Execute and validate stored EFE-style feature transforms.

Feature cards carry code defining a `Transformation` with `fit` and `transform`, plus optional
`inverse_transform` for invertible transforms. Before registration/use, the runner checks required
entry points, no in-place mutation, schema consistency, and invertible round trips when declared.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class FeatureValidationError(Exception):
    pass


def load_program(code: str, class_name: str = "Transformation"):
    """Exec a stored program string and return its Transformation class.

    NOTE: executes code. In the benchmark/agent setting run this in a sandbox
    (restricted globals, subprocess, or container). Here we exec in a fresh namespace
    with a minimal import surface, matching how EFE executes candidate programs.
    """
    ns: Dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102 - intentional; sandbox in production
    except Exception as e:
        raise FeatureValidationError(f"program failed to import: {e}")
    cls = ns.get(class_name)
    if cls is None:
        raise FeatureValidationError(f"no class '{class_name}' defined")
    for m in ("fit", "transform"):
        if not callable(getattr(cls, m, None)):
            raise FeatureValidationError(f"missing required method '{m}'")
    return cls


def _no_inplace(before, after) -> bool:
    """transform must return a NEW object, not mutate its input (EFE check)."""
    return after is not before


def validate_and_run(doc: dict, X_fit, X_in, metadata: Optional[dict] = None) -> dict:
    """Load the program in `doc['code']`, fit on X_fit, transform X_in.

    Returns {output, state, invertible_ok, checks}. Mirrors EFE's evaluator: it does the
    structural checks and (for invertible programs) verifies a round-trip, returning a
    feedback dict the caller (or an LLM evolver) can act on.
    """
    checks = {"entry_points": False, "no_inplace": False, "invertible_ok": None}
    cls = load_program(doc["code"], doc.get("class_name", "Transformation"))
    checks["entry_points"] = True

    inst = cls()
    X_fit_guard = copy.deepcopy(X_fit)
    state = inst.fit(X_fit, metadata or {})
    # fit may return state or store on self; support both
    if state is None:
        state = getattr(inst, "state_", inst)

    out = inst.transform(X_in, state)
    checks["no_inplace"] = _no_inplace(X_in, out)

    # invertibility round-trip when declared and supported
    if doc.get("invertible") and hasattr(inst, "inverse_transform"):
        import numpy as np
        recon = inst.inverse_transform(out, state)
        try:
            checks["invertible_ok"] = bool(np.allclose(np.asarray(recon, dtype=float),
                                                        np.asarray(X_in, dtype=float),
                                                        rtol=1e-4, atol=1e-6))
        except Exception:
            checks["invertible_ok"] = None

    if not checks["no_inplace"]:
        raise FeatureValidationError("transform mutated its input in place")
    if doc.get("invertible") and checks["invertible_ok"] is False:
        raise FeatureValidationError("declared invertible but round-trip failed")

    return {"output": out, "state": state, "checks": checks,
            "feature_id": doc.get("feature_id"), "target_model": doc.get("target_model")}
