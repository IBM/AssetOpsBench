"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Tools:
  get_failure_modes                    – READ the (partial) failure-mode list for an asset from CouchDB
  generate_failure_modes               – GENERATE failure modes via the LLM (when DB is missing/partial)
  add_failure_modes                    – WRITE: persist/augment a class's failure modes in CouchDB
  generate_failure_mode_sensor_mapping – GENERATE the bidirectional FM↔sensor relevancy via the LLM

Failure modes live in CouchDB (collection "failure_mode", doctype "failure_mode", one doc per asset
class, seeded from failure_modes.yaml). Coverage is NOT exhaustive: docs carry `exhaustive: false`, so
get_failure_modes returns what is known and signals when generation is needed. Retrieval (get_*) and
generation (generate_*) are separate by design.

LLM backend is configured via FMSR_MODEL_ID (default: watsonx/meta-llama/llama-3-3-70b-instruct).
CouchDB via COUCHDB_URL / FAILURE_MODE_DBNAME / CATALOG_DBNAME / COUCHDB_USERNAME / COUCHDB_PASSWORD.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── CouchDB stores ────────────────────────────────────────────────────────────
# Under AssetOpsBench's loader, database name == collection key, so the FMSR data is two separate
# databases (loaded from the manifest, no seed): 'failure_mode' and 'catalog'. We open one handle
# each — there is no single 'fmsr' database.

COUCHDB_URL = os.environ.get("COUCHDB_URL")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
FAILURE_MODE_DBNAME = os.environ.get("FAILURE_MODE_DBNAME", "failure_mode")
CATALOG_DBNAME = os.environ.get("CATALOG_DBNAME", "catalog")


def _connect(dbname):
    try:
        h = couchdb3.Database(dbname, url=COUCHDB_URL, user=COUCHDB_USERNAME, password=COUCHDB_PASSWORD)
        logger.info("Connected to CouchDB: %s", dbname)
        return h
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to connect to CouchDB '%s': %s", dbname, e)
        return None


fm_db = _connect(FAILURE_MODE_DBNAME)        # failure_mode docs
catalog_db = _connect(CATALOG_DBNAME)        # catalog docs


def _asset_key(asset_name: str) -> str:
    """Normalise an asset name to a class key: strip digits, trim, lowercase ('Pump 1' -> 'pump')."""
    return re.sub(r"\d+", "", asset_name or "").strip().lower()


# ── Prompt templates ──────────────────────────────────────────────────────────

_ASSET2FM_PROMPT = (
    "What are different failure modes for asset {asset_name}?\n"
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
)

_ASSET2FM_EXTEND_PROMPT = (
    "The asset {asset_name} already has these known failure modes:\n{known}\n\n"
    "List ADDITIONAL failure modes for {asset_name} that are NOT already in the list above. "
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name."
)

_RELEVANCY_PROMPT = (
    "For the asset {asset_name}, if the failure {failure_mode} occurs, "
    "can sensor {sensor} help monitor or detect the failure for {asset_name}?\n"
    "Provide the answer in the first line and reason in the second line. "
    "If the answer is Yes, provide the temporal behaviour of the sensor "
    "when the failure occurs in the third line."
)


# ── Output parsers ────────────────────────────────────────────────────────────

def _parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


def _parse_relevancy(text: str) -> dict:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("yes"):
        answer = "Yes"
    elif lines and lines[0].lower().startswith("no"):
        answer = "No"
    else:
        answer = "Unknown"
    reason = lines[1] if len(lines) >= 2 else "Unknown"
    temporal = lines[2] if (answer == "Yes" and len(lines) >= 3) else "Unknown"
    return {"answer": answer, "reason": reason, "temporal_behavior": temporal}


# ── LLM backend (lazy init; graceful degradation if creds are absent) ─────────

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"
_MAX_RETRIES = 3
_MODEL_ID = os.environ.get("FMSR_MODEL_ID", _DEFAULT_MODEL_ID)


def _build_llm():
    from llm import LiteLLMBackend

    if _MODEL_ID.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for WatsonX: {missing}")
    else:
        missing = [v for v in ("LITELLM_API_KEY", "LITELLM_BASE_URL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for LiteLLM: {missing}")
    return LiteLLMBackend(_MODEL_ID)


try:
    _llm = _build_llm()
    _llm_available = True
except Exception as _e:  # noqa: BLE001
    logger.warning("LLM unavailable (generate_* tools disabled): %s", _e)
    _llm = None
    _llm_available = False


_asset2fm_cache: dict[str, list[str]] = {}


def _call_asset2fm(asset_name: str) -> list[str]:
    if asset_name in _asset2fm_cache:
        return _asset2fm_cache[asset_name]
    prompt = _ASSET2FM_PROMPT.format(asset_name=asset_name)
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            result = _parse_numbered_list(_llm.generate(prompt))
            _asset2fm_cache[asset_name] = result
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


def _call_asset2fm_extend(asset_name: str, known: list[str]) -> list[str]:
    """Ask the LLM for ADDITIONAL failure modes given the already-known ones. Retries up to
    _MAX_RETRIES. Not cached (depends on the known list)."""
    prompt = _ASSET2FM_EXTEND_PROMPT.format(
        asset_name=asset_name, known="\n".join(f"- {k}" for k in known)
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_numbered_list(_llm.generate(prompt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


def _call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    prompt = _RELEVANCY_PROMPT.format(asset_name=asset_name, failure_mode=failure_mode, sensor=sensor)
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_relevancy(_llm.generate(prompt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


# ── Result models ─────────────────────────────────────────────────────────────

class ErrorResult(BaseModel):
    error: str


class FailureModesResult(BaseModel):
    asset_name: str
    failure_modes: List[str]
    exhaustive: bool = False          # the stored list is not claimed to be complete
    source: Optional[str] = None      # provenance: ISO / curated / LLM:<model>


class GenerateFailureModesResult(BaseModel):
    asset_name: str
    known: List[str]                  # pre-existing modes used as context (from DB or provided)
    generated: List[str]              # newly generated modes NOT already in `known`
    failure_modes: List[str]          # known + generated (the extended list)
    source: str                       # LLM:<model>
    message: str


class AddFailureModesResult(BaseModel):
    asset_class: str
    added: List[str]                  # newly inserted (not previously present)
    total: int                        # total after the write
    exhaustive: bool
    source: Optional[str]
    message: str


class CatalogResult(BaseModel):
    kind: str                         # "sensor" | "failure_mode"
    scenario_id: Optional[str]        # which scope was served (None = global default)
    total: int
    items: List[str]
    source: Optional[str]
    message: str


class RelevancyEntry(BaseModel):
    asset_name: str
    failure_mode: str
    sensor: str
    relevancy_answer: str
    relevancy_reason: str
    temporal_behavior: str


class MappingMetadata(BaseModel):
    asset_name: str
    failure_modes: List[str]
    sensors: List[str]


class FailureModeSensorMappingResult(BaseModel):
    metadata: MappingMetadata
    fm2sensor: Dict[str, List[str]]
    sensor2fm: Dict[str, List[str]]
    full_relevancy: List[RelevancyEntry]


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "fmsr",
    instructions=(
        "Failure mode and sensor reasoning. get_failure_modes READS a class's (possibly partial) "
        "failure modes from CouchDB; generate_failure_modes GENERATES them via the LLM when the DB "
        "is missing or incomplete (exhaustive=false); add_failure_modes WRITES them back; "
        "generate_failure_mode_sensor_mapping GENERATES which sensors can detect each failure. "
        "get_sensor_catalog / get_failure_mode_catalog READ the global (class-independent) reference "
        "lists from the loaded catalog dataset (scenario-scoped if a scenario_id catalog was loaded); "
        "if no catalog collection is loaded they report that none is available."
    ),
)


@mcp.tool(title="Get Failure Modes")
def get_failure_modes(asset_name: str) -> Union[FailureModesResult, ErrorResult]:
    """READ the known failure modes for an asset from CouchDB (collection 'failure_mode'). The list
    may be partial: check `exhaustive` — if false, call generate_failure_modes to supplement.
    Does NOT call the LLM. Returns an error if the class is not in the DB."""
    key = _asset_key(asset_name)
    if not key or key == "none":
        return ErrorResult(error="asset_name is required")
    if not fm_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        res = fm_db.find({"asset_class": key}, limit=1)
        docs = res["docs"]
        if not docs:
            return ErrorResult(
                error=f"no failure_mode record for '{key}' in DB; try generate_failure_modes"
            )
        d = docs[0]
        return FailureModesResult(
            asset_name=asset_name,
            failure_modes=d.get("failure_modes", []),
            exhaustive=d.get("exhaustive", False),
            source=d.get("source"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


def _known_failure_modes(asset_name: str) -> List[str]:
    """Current (partial) failure modes stored for the asset's class, [] if none / no DB."""
    if not fm_db:
        return []
    try:
        r = fm_db.find({"asset_class": _asset_key(asset_name)}, limit=1)
        return r["docs"][0].get("failure_modes", []) if r["docs"] else []
    except Exception:  # noqa: BLE001
        return []


@mcp.tool(title="Generate Failure Modes")
def generate_failure_modes(
    asset_name: str, known: Optional[List[str]] = None
) -> Union[GenerateFailureModesResult, ErrorResult]:
    """GENERATE failure modes for an asset via the LLM, EXTENDING the known (partial) list. The DB
    list is usually not exhaustive, so this asks the LLM for ADDITIONAL modes beyond what is already
    known. If `known` is omitted, the current DB list for the class is used as context; if there is
    no known list, it generates from scratch. Generated modes already present in `known` are dropped.
    Nothing is persisted — call add_failure_modes to save the new ones."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")
    base = known if known is not None else _known_failure_modes(asset_name)
    base = [k.strip() for k in base if k and k.strip()]
    try:
        raw = _call_asset2fm_extend(asset_name, base) if base else _call_asset2fm(asset_name)
        seen = {k.lower() for k in base}
        new: List[str] = []
        for g in raw:
            if g and g.lower() not in seen:
                seen.add(g.lower())
                new.append(g)
        return GenerateFailureModesResult(
            asset_name=asset_name, known=base, generated=new, failure_modes=base + new,
            source=f"LLM:{_MODEL_ID}",
            message=(f"generated {len(new)} new failure mode(s) extending {len(base)} known "
                     f"({len(base) + len(new)} total); call add_failure_modes to persist."),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Add Failure Modes")
def add_failure_modes(
    asset_class: str,
    failure_modes: List[str],
    exhaustive: bool = False,
    source: Optional[str] = None,
) -> Union[AddFailureModesResult, ErrorResult]:
    """WRITE: persist/augment the failure modes for a class in CouchDB. Merges with any existing list
    (union, de-duplicated). Use to save generated or curated modes so future get_failure_modes calls
    return them. Set exhaustive=true only if the list is now believed complete."""
    key = _asset_key(asset_class)
    if not key:
        return ErrorResult(error="asset_class is required")
    if not failure_modes:
        return ErrorResult(error="failure_modes list is required")
    if not fm_db:
        return ErrorResult(error="CouchDB not connected")
    doc_id = f"fm:{key}"
    try:
        try:
            doc = fm_db.get(doc_id)
        except Exception:  # noqa: BLE001
            doc = None
        existing = set(doc.get("failure_modes", [])) if doc else set()
        incoming = {fm.strip() for fm in failure_modes if fm and fm.strip()}
        added = sorted(incoming - existing)
        merged = sorted(existing | incoming)
        new_source = source or (doc.get("source") if doc else None) or "user"
        if doc:
            doc["failure_modes"] = merged
            doc["exhaustive"] = exhaustive
            doc["source"] = new_source
            fm_db.save(doc)
        else:
            fm_db.save({
                "_id": doc_id, "asset_class": key,
                "failure_modes": merged, "exhaustive": exhaustive, "source": new_source,
            })
        return AddFailureModesResult(
            asset_class=key, added=added, total=len(merged), exhaustive=exhaustive,
            source=new_source,
            message=f"added {len(added)} new failure mode(s) to '{key}' ({len(merged)} total).",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("add_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


def _read_catalog(kind: str, scenario_id: Optional[str]):
    """Return the catalog doc for (kind): the scenario-scoped doc if scenario_id is given and exists,
    otherwise the global default (scenario_id = null). None if absent / no DB."""
    if not catalog_db:
        return None
    if scenario_id:
        r = catalog_db.find({"doctype": "catalog", "kind": kind, "scenario_id": scenario_id}, limit=1)
        if r["docs"]:
            return r["docs"][0]
    r = catalog_db.find({"doctype": "catalog", "kind": kind, "scenario_id": None}, limit=1)
    return r["docs"][0] if r["docs"] else None


def _catalog_result(kind: str, scenario_id: Optional[str]) -> Union[CatalogResult, ErrorResult]:
    if not catalog_db:
        return ErrorResult(error="CouchDB not connected")
    doc = _read_catalog(kind, scenario_id)
    if not doc:
        return ErrorResult(error=f"no catalog information available for kind '{kind}'")
    items = doc.get("items", [])
    served = doc.get("scenario_id")
    return CatalogResult(
        kind=kind, scenario_id=served, total=len(items), items=items, source=doc.get("source"),
        message=f"{len(items)} {kind} items "
                f"({'scenario ' + served if served else 'global default'} catalog).",
    )


@mcp.tool(title="Get Sensor Catalog")
def get_sensor_catalog(scenario_id: Optional[str] = None) -> Union[CatalogResult, ErrorResult]:
    """READ the catalog of all potential sensors / monitoring parameters (class-independent reference
    list). If scenario_id is given and a scenario-scoped catalog exists, it is returned; otherwise the
    global default. Useful as the candidate-sensor input to generate_failure_mode_sensor_mapping."""
    return _catalog_result("sensor", scenario_id)


@mcp.tool(title="Get Failure Mode Catalog")
def get_failure_mode_catalog(scenario_id: Optional[str] = None) -> Union[CatalogResult, ErrorResult]:
    """READ the catalog of all potential failure modes (class-independent reference list). Scenario-
    scoped if scenario_id matches a registered catalog, else the global default."""
    return _catalog_result("failure_mode", scenario_id)


@mcp.tool(title="Generate Failure Mode Sensor Mapping")
def generate_failure_mode_sensor_mapping(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
) -> Union[FailureModeSensorMappingResult, ErrorResult]:
    """GENERATE, for each (failure_mode, sensor) pair, whether the sensor can detect the failure
    (one LLM call per pair). Returns a bidirectional mapping (fm→sensors, sensor→fms) plus per-pair
    relevancy details. Keep both lists small (e.g. ≤5 failure modes, ≤10 sensors) to bound runtime."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")
    if not failure_modes:
        return ErrorResult(error="failure_modes list is required")
    if not sensors:
        return ErrorResult(error="sensors list is required")
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    full_relevancy: List[RelevancyEntry] = []
    fm2sensor: Dict[str, List[str]] = {}
    sensor2fm: Dict[str, List[str]] = {}
    try:
        pairs = [(s, fm) for s in sensors for fm in failure_modes]
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_call_relevancy, asset_name, fm, s): (s, fm) for s, fm in pairs
            }
            for future in as_completed(futures):
                s, fm = futures[future]
                gen = future.result()
                full_relevancy.append(RelevancyEntry(
                    asset_name=asset_name, failure_mode=fm, sensor=s,
                    relevancy_answer=gen["answer"], relevancy_reason=gen["reason"],
                    temporal_behavior=gen["temporal_behavior"],
                ))
                if "yes" in gen["answer"].lower():
                    fm2sensor.setdefault(fm, []).append(s)
                    sensor2fm.setdefault(s, []).append(fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("_call_relevancy failed: %s", exc)
        return ErrorResult(error=str(exc))

    return FailureModeSensorMappingResult(
        metadata=MappingMetadata(asset_name=asset_name, failure_modes=failure_modes, sensors=sensors),
        fm2sensor=fm2sensor, sensor2fm=sensor2fm, full_relevancy=full_relevancy,
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()