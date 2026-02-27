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

import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Hardcoded asset data ──────────────────────────────────────────────────────

_FAILURE_MODES_FILE = Path(__file__).parent / "failure_modes.yaml"
with _FAILURE_MODES_FILE.open() as _f:
    _ASSET_FAILURE_MODES: dict[str, list[str]] = yaml.safe_load(_f)


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

def _call_asset2fm(asset_name: str) -> list[str]:
    """Query the LLM for failure modes of an asset. Retries up to _MAX_RETRIES times."""
    prompt = _ASSET2FM_PROMPT.format(asset_name=asset_name)
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_numbered_list(_llm.generate(prompt))
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
            return _parse_relevancy(_llm.generate(prompt))
        except Exception as exc:
            last_exc = exc
    raise last_exc


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

mcp = FastMCP("FMSRAgent")


@mcp.tool()
def get_failure_modes(asset_name: str) -> Union[FailureModesResult, ErrorResult]:
    """Returns a list of known failure modes for the given asset.
    For chillers and AHUs returns a curated list. For other assets queries the LLM."""
    asset_key = re.sub(r"\d+", "", asset_name).strip().lower()
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


@mcp.tool()
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

    try:
        for s in sensors:
            for fm in failure_modes:
                gen = _call_relevancy(asset_name, fm, s)
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


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
