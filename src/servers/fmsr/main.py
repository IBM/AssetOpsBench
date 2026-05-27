"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Exposes two tools:
  get_failure_modes               – lists failure modes for an asset
  get_failure_mode_sensor_mapping – returns bidirectional FM↔sensor relevancy mapping

For chillers and AHUs get_failure_modes returns a curated hardcoded list.
For any other asset type the LLM is queried as a fallback.
The mapping tool always calls the LLM to determine per-pair relevancy.

LLM backend is configured via the FMSR_MODEL_ID environment variable
(default: ``watsonx/meta-llama/llama-3-3-70b-instruct``).  Any model string
supported by litellm works — the provider is encoded in the prefix.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .models import (
    DGAInterpretationResult,
    HealthIndexResult,
    WindingTemperatureResult,
    LoadProfileResult,
)
from .prompt_templates import (
    _INTERPRET_DGA_PROMPT,
    _ASSESS_WINDING_PROMPT,
    _ASSESS_LOAD_PROMPT,
)

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Hardcoded asset data ──────────────────────────────────────────────────────

_FAILURE_MODES_FILE = Path(__file__).parent / "failure_modes.yaml"
with _FAILURE_MODES_FILE.open() as _f:
    _ASSET_FAILURE_MODES: dict[str, list[str]] = yaml.safe_load(_f)

_ASSET_FAILURE_MODE_ALIASES = {
    "transformer": "smart grid transformer",
}

# ── Prompt templates ──────────────────────────────────────────────────────────

_ASSET2FM_PROMPT = (
    "What are different failure modes for asset {asset_name}?\n"
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
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
    """Parse a numbered list response into a plain list of strings."""
    items = []
    for line in text.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


def _normalize_asset_key(asset_name: str) -> str:
    """Normalize an asset name for curated failure-mode lookup."""
    normalized = re.sub(r"\d+", "", asset_name or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized


def _resolve_failure_mode_asset_key(asset_name: str) -> str:
    """Resolve an asset name to a curated failure-mode key when possible."""
    asset_key = _normalize_asset_key(asset_name)
    if asset_key in _ASSET_FAILURE_MODES:
        return asset_key
    if asset_key in _ASSET_FAILURE_MODE_ALIASES:
        return _ASSET_FAILURE_MODE_ALIASES[asset_key]

    for known_key in _ASSET_FAILURE_MODES:
        if asset_key and (asset_key in known_key or known_key in asset_key):
            return known_key
    return asset_key


def _parse_relevancy(text: str) -> dict:
    """Parse a 3-line relevancy response into {answer, reason, temporal_behavior}."""
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

def _parse_dga_response(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return {
        "fault_type": result.get("Fault Type", "Unknown"),
        "r1": float(result.get("R1 (CH4/H2)", 0) or 0),
        "r2": float(result.get("R2 (C2H2/C2H4)", 0) or 0),
        "r3": float(result.get("R3 (C2H4/C2H6)", 0) or 0),
        "code": result.get("Code (R1,R2,R3)", "Unknown"),
        "confidence": result.get("Confidence", "Unknown"),
        "reasoning": result.get("Reasoning", ""),
        "recommended_action": result.get("Recommended Action", ""),
    }


def _parse_winding_response(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return {
        "thermal_status": result.get("Thermal Status", "Unknown"),
        "hot_spot_rise_c": float(result.get("Hot-Spot Rise (C)", 0) or 0),
        "ageing_rate": float(result.get("Ageing Rate", 1.0) or 1.0),
        "alarm_active": result.get("Alarm Active", "No") == "Yes",
        "trip_active": result.get("Trip Active", "No") == "Yes",
        "risk_level": result.get("Risk Level", "Unknown"),
        "reasoning": result.get("Reasoning", ""),
        "recommended_action": result.get("Recommended Action", ""),
    }


def _parse_load_response(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return {
        "load_mva": float(result.get("Load MVA", 0) or 0),
        "load_factor_pct": float(result.get("Load Factor (%)", 0) or 0),
        "loading_status": result.get("Loading Status", "Unknown"),
        "current_imbalance_pct": float(result.get("Current Imbalance (%)", 0) or 0),
        "neutral_current_flag": result.get("Neutral Current Flag", "No") == "Yes",
        "reasoning": result.get("Reasoning", ""),
        "recommended_action": result.get("Recommended Action", ""),
    }


# ── LLM backend (lazy init; graceful degradation if creds are absent) ─────────

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"
_MAX_RETRIES = 3


def _build_llm():
    from llm import LiteLLMBackend

    model_id = os.environ.get("FMSR_MODEL_ID", _DEFAULT_MODEL_ID)
    if model_id.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for WatsonX: {missing}")
    else:
        missing = [v for v in ("LITELLM_API_KEY", "LITELLM_BASE_URL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for LiteLLM: {missing}")
    return LiteLLMBackend(model_id)


try:
    _llm = _build_llm()
    _llm_available = True
except Exception as _e:
    logger.warning("LLM unavailable (FMSR will use curated data only): %s", _e)
    _llm = None
    _llm_available = False


# ── LLM call helpers with retry ───────────────────────────────────────────────

_asset2fm_cache: dict[str, list[str]] = {}


def _call_asset2fm(asset_name: str) -> list[str]:
    """Query the LLM for failure modes of an asset. Retries up to _MAX_RETRIES times.
    Results are cached to avoid redundant LLM calls for the same asset."""
    if asset_name in _asset2fm_cache:
        return _asset2fm_cache[asset_name]

    prompt = _ASSET2FM_PROMPT.format(asset_name=asset_name)
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            result = _parse_numbered_list(_llm.generate(prompt).text)
            _asset2fm_cache[asset_name] = result
            return result
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    """Query the LLM for FM↔sensor relevancy. Retries up to _MAX_RETRIES times."""
    prompt = _RELEVANCY_PROMPT.format(
        asset_name=asset_name, failure_mode=failure_mode, sensor=sensor
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_relevancy(_llm.generate(prompt).text)
        except Exception as exc:
            last_exc = exc
    raise last_exc

def _call_dga(
    asset_name: str,
    hydrogen: float,
    methane: float,
    acetylene: float,
    ethylene: float,
    ethane: float,
) -> dict:
    prompt = _INTERPRET_DGA_PROMPT.format(
        asset_name=asset_name,
        hydrogen=hydrogen,
        methane=methane,
        acetylene=acetylene,
        ethylene=ethylene,
        ethane=ethane,
    )

    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            raw = _llm.generate(prompt).text
            return _parse_dga_response(raw)
        except Exception as exc:
            last_exc = exc
    raise last_exc

def _call_winding(
    asset_name: str, wti: float, oti: float, ati: float, oti_a: int, oti_t: int
) -> dict:
    prompt = _ASSESS_WINDING_PROMPT.format(
        asset_name=asset_name,
        wti=wti,
        oti=oti,
        ati=ati,
        oti_a=oti_a,
        oti_t=oti_t,
    )

    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            raw = _llm.generate(prompt).text
            return _parse_winding_response(raw)
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _call_load(
    asset_name: str,
    vl1: float,
    vl2: float,
    vl3: float,
    il1: float,
    il2: float,
    il3: float,
    vl12: float,
    vl23: float,
    vl31: float,
    inut: float,
    rated_mva: float,
) -> dict:
    prompt = _ASSESS_LOAD_PROMPT.format(
        asset_name=asset_name,
        vl1=vl1,
        vl2=vl2,
        vl3=vl3,
        il1=il1,
        il2=il2,
        il3=il3,
        vl12=vl12,
        vl23=vl23,
        vl31=vl31,
        inut=inut,
        rated_mva=rated_mva,
    )

    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            raw = _llm.generate(prompt).text
            return _parse_load_response(raw)
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _call_predict_health_index(
    hydrogen: float, oxygen: float, nitrogen: float,
    methane: float, co: float, co2: float,
    ethylene: float, ethane: float, acetylene: float,
    dbds: float, power_factor: float, interfacial_v: float,
    dielectric_rigidity: float, water_content: float,
) -> float:
    import pickle
    import numpy as np

    base_path = Path(__file__).parent / "artifacts"

    with (base_path / "health_index_model.pkl").open("rb") as f:
        model = pickle.load(f)

    with (base_path / "health_index_scalers.pkl").open("rb") as f:
        scaler_X = pickle.load(f)["scaler_X"]

    feature_values = np.array(
        [[
            hydrogen,
            oxygen,
            nitrogen,
            methane,
            co,
            co2,
            ethylene,
            ethane,
            acetylene,
            dbds,
            power_factor,
            interfacial_v,
            dielectric_rigidity,
            water_content,
        ]]
    )

    scaled = scaler_X.transform(feature_values)
    score = model.predict(scaled)[0]
    return float(score)


# ── Result models ─────────────────────────────────────────────────────────────

class ErrorResult(BaseModel):
    error: str


class FailureModesResult(BaseModel):
    asset_name: str
    failure_modes: List[str]


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

mcp = FastMCP("fmsr", instructions="Failure mode and sensor reasoning: get failure modes for assets and determine which sensors can detect each failure.")


@mcp.tool(title="Get Failure Modes")
def get_failure_modes(asset_name: str) -> Union[FailureModesResult, ErrorResult]:
    """Returns a list of known failure modes for the given asset.
    For chillers and AHUs returns a curated list. For other assets queries the LLM."""
    asset_key = _resolve_failure_mode_asset_key(asset_name)
    if not asset_key or asset_key == "none":
        return ErrorResult(error="asset_name is required")

    if asset_key in _ASSET_FAILURE_MODES:
        return FailureModesResult(
            asset_name=asset_name,
            failure_modes=_ASSET_FAILURE_MODES[asset_key],
        )

    if not _llm_available:
        return ErrorResult(error="LLM unavailable and asset not in local database")

    try:
        result = _call_asset2fm(asset_name)
        return FailureModesResult(asset_name=asset_name, failure_modes=result)
    except Exception as exc:
        logger.error("_call_asset2fm failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Failure Mode Sensor Mapping")
def get_failure_mode_sensor_mapping(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
) -> Union[FailureModeSensorMappingResult, ErrorResult]:
    """For each (failure_mode, sensor) pair determines whether the sensor can detect
    the failure. Returns a bidirectional mapping (fm→sensors, sensor→fms) plus
    the full per-pair relevancy details.

    Note: one LLM call is made per (failure_mode, sensor) pair sequentially.
    Keep both lists small (e.g. ≤5 failure modes, ≤10 sensors) to avoid long
    runtimes. For a chiller with 7 failure modes and 20+ sensors the call will
    take several minutes."""
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

    total_calls = len(failure_modes) * len(sensors)
    logger.info(
        "Starting FM-sensor mapping for asset '%s' with %d failure modes and %d sensors (%d total calls)",
        asset_name,
        len(failure_modes),
        len(sensors),
        total_calls,
    )

    try:
        pairs = [(s, fm) for s in sensors for fm in failure_modes]
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_call_relevancy, asset_name, fm, s): (s, fm)
                for s, fm in pairs
            }
            for future in as_completed(futures):
                s, fm = futures[future]
                gen = future.result()
                entry = RelevancyEntry(
                    asset_name=asset_name,
                    failure_mode=fm,
                    sensor=s,
                    relevancy_answer=gen["answer"],
                    relevancy_reason=gen["reason"],
                    temporal_behavior=gen["temporal_behavior"],
                )
                full_relevancy.append(entry)
                if "yes" in gen["answer"].lower():
                    fm2sensor.setdefault(fm, []).append(s)
                    sensor2fm.setdefault(s, []).append(fm)
    except Exception as exc:
        logger.error("_call_relevancy failed: %s", exc)
        return ErrorResult(error=str(exc))

    return FailureModeSensorMappingResult(
        metadata=MappingMetadata(
            asset_name=asset_name,
            failure_modes=failure_modes,
            sensors=sensors,
        ),
        fm2sensor=fm2sensor,
        sensor2fm=sensor2fm,
        full_relevancy=full_relevancy,
    )


@mcp.tool()
def interpret_dga(
    asset_name: str,
    hydrogen: float,
    methane: float,
    acetylene: float,
    ethylene: float,
    ethane: float,
) -> Union[DGAInterpretationResult, ErrorResult]:
    """Interpret dissolved gas analysis readings for a transformer."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")

    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    try:
        parsed = _call_dga(asset_name, hydrogen, methane, acetylene, ethylene, ethane)
        return DGAInterpretationResult(
            asset_name=asset_name,
            **parsed,
        )
    except Exception as exc:
        logger.error("_call_dga failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool()
def assess_winding_temperature(
    asset_name: str,
    wti: float,
    oti: float,
    ati: float,
    oti_a: int,
    oti_t: int,
) -> Union[WindingTemperatureResult, ErrorResult]:
    """Assess transformer winding temperature condition."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")

    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    try:
        parsed = _call_winding(asset_name, wti, oti, ati, oti_a, oti_t)
        return WindingTemperatureResult(
            asset_name=asset_name,
            **parsed,
        )
    except Exception as exc:
        logger.error("_call_winding failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool()
def assess_load_profile(
    asset_name: str,
    vl1: float,
    vl2: float,
    vl3: float,
    il1: float,
    il2: float,
    il3: float,
    vl12: float,
    vl23: float,
    vl31: float,
    inut: float,
    rated_mva: float,
) -> Union[LoadProfileResult, ErrorResult]:
    """Assess transformer loading condition."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")

    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    try:
        parsed = _call_load(
            asset_name,
            vl1,
            vl2,
            vl3,
            il1,
            il2,
            il3,
            vl12,
            vl23,
            vl31,
            inut,
            rated_mva,
        )
        return LoadProfileResult(
            asset_name=asset_name,
            **parsed,
        )
    except Exception as exc:
        logger.error("_call_load failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool()
def predict_health_index(
    asset_name: str,
    hydrogen: float,
    oxygen: float,
    nitrogen: float,
    methane: float,
    co: float,
    co2: float,
    ethylene: float,
    ethane: float,
    acetylene: float,
    dbds: float,
    power_factor: float,
    interfacial_v: float,
    dielectric_rigidity: float,
    water_content: float,
) -> Union[HealthIndexResult, ErrorResult]:
    """Predict a transformer health index from condition data."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")

    try:
        score = _call_predict_health_index(
            hydrogen,
            oxygen,
            nitrogen,
            methane,
            co,
            co2,
            ethylene,
            ethane,
            acetylene,
            dbds,
            power_factor,
            interfacial_v,
            dielectric_rigidity,
            water_content,
        )

        if score >= 85:
            condition = "Very Good"
        elif score >= 70:
            condition = "Good"
        elif score >= 50:
            condition = "Fair"
        elif score >= 30:
            condition = "Poor"
        else:
            condition = "Very Poor"

        return HealthIndexResult(
            asset_name=asset_name,
            health_index=score,
            condition=condition,
        )

    except Exception as exc:
        logger.error("_call_predict_health_index failed: %s", exc)
        return ErrorResult(error=str(exc))


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
