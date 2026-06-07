"""Optional per-collection transforms — the escape hatch for the generic loader.

Define a function named after a collection key and the loader applies it to each
document (after generic parsing + dotted-header nesting, before the _id is computed).
A transform takes a doc dict and returns a doc dict (or None to keep the same one).

Most collections need nothing here — iot / vibration / alert / event load straight
from collections.json. work_order is the one that needs CSV value typing (CSV cells
arrive as strings) and the Maximo ``type`` discriminator the design doc validates.
"""

import json

# Work-order CSV columns that must be typed (CSV gives everything as strings).
_WO_INT = ("wopriority", "taskid")
_WO_FLOAT = (
    "estlabhrs", "actlabhrs", "estlabcost", "actlabcost", "estmatcost", "actmatcost",
    "estservcost", "actservcost", "esttoolcost", "acttoolcost", "estatapprtotalcost",
    "esttotalcost", "acttotalcost",
)
_WO_EVIDENCE_FLOAT = ("anomaly_score", "threshold", "observed_value")  # nested under aob_source.evidence


def workorder(doc):
    """Type a work-order doc and stamp the Maximo ``type`` (used by the type indexes)."""
    doc.setdefault("type", "workorder")

    for f in _WO_INT:
        if isinstance(doc.get(f), str):
            doc[f] = int(float(doc[f]))
    for f in _WO_FLOAT:
        if isinstance(doc.get(f), str):
            doc[f] = float(doc[f])

    if isinstance(doc.get("wplabor"), str):          # JSON-string column → list
        doc["wplabor"] = json.loads(doc["wplabor"])

    evidence = doc.get("aob_source", {}).get("evidence")
    if isinstance(evidence, dict):
        for f in _WO_EVIDENCE_FLOAT:
            if isinstance(evidence.get(f), str):
                evidence[f] = float(evidence[f])

    return doc