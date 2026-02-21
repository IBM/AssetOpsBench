"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Exposes two tools:
  get_failure_modes               – lists failure modes for an asset
  get_failure_mode_sensor_mapping – returns bidirectional FM↔sensor relevancy mapping

For chillers and AHUs get_failure_modes returns a curated hardcoded list.
For any other asset type the LLM is queried as a fallback.
The mapping tool always calls the LLM to determine per-pair relevancy.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Union

import yaml

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
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


# ── Output parsers ────────────────────────────────────────────────────────────

class _RelevancyParser(JsonOutputParser):
    """Parses a 3-line LLM response into {answer, reason, temporal_behavior}."""

    def parse_result(self, result, *, partial: bool = False):
        text = result[0].text.strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith("yes"):
            answer = "Yes"
        elif lines and lines[0].lower().startswith("no"):
            answer = "No"
        else:
            answer = "Unknown"
        reason = lines[1] if len(lines) >= 2 else "Unknown"
        temporal = lines[2] if (answer == "Yes" and len(lines) >= 3) else "Unknown"
        return {"answer": answer, "reason": reason, "temporal_behavior": temporal}


# ── Prompts ───────────────────────────────────────────────────────────────────

_asset2fm_prompt = ChatPromptTemplate.from_template(
    "What are different failure modes for asset {asset_name}?\n"
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
)

_relevancy_prompt = ChatPromptTemplate.from_template(
    "For the asset {asset_name}, if the failure {failure_mode} occurs, "
    "can sensor {sensor} help monitor or detect the failure for {asset_name}?\n"
    "Provide the answer in the first line and reason in the second line. "
    "If the answer is Yes, provide the temporal behaviour of the sensor "
    "when the failure occurs in the third line."
)


# ── LLM + chains (lazy init; graceful degradation if creds are absent) ────────

def _build_chains():
    # ibm_watsonx_ai's StrEnum subclass passes member args to super().__init__(),
    # which Python 3.14's object.__init__ rejects. Patch before the full import
    # chain reaches ibm_watsonx_ai.utils.autoai.enums where the class is defined.
    import sys
    if "ibm_watsonx_ai.utils.autoai.enums" not in sys.modules:
        import ibm_watsonx_ai.utils.utils as _wx_utils
        _wx_utils.StrEnum.__init__ = lambda self, *a, **k: None

    from langchain_ibm import ChatWatsonx

    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=os.environ["WATSONX_URL"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        api_key=os.environ["WATSONX_APIKEY"],
        params={"max_tokens": 10000, "temperature": 0.0},
    )
    llm_with_retry = llm.with_retry(stop_after_attempt=3)
    return (
        _asset2fm_prompt | llm_with_retry | NumberedListOutputParser(),
        _relevancy_prompt | llm | _RelevancyParser(),
    )


try:
    _asset2fm_chain, _relevancy_chain = _build_chains()
    _llm_available = True
except Exception as _e:
    logger.error("WatsonX LLM unavailable: %s", _e)
    _asset2fm_chain = _relevancy_chain = None
    _llm_available = False


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
        result = _asset2fm_chain.invoke({"asset_name": asset_name})
        return FailureModesResult(asset_name=asset_name, failure_modes=result)
    except Exception as exc:
        logger.error("asset2fm_chain failed: %s", exc)
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

    batches = [
        {"asset_name": asset_name, "failure_mode": fm, "sensor": s}
        for s in sensors
        for fm in failure_modes
    ]

    try:
        generations = _relevancy_chain.batch(batches, config={"max_concurrency": 1})
    except Exception as exc:
        logger.error("relevancy_chain.batch failed: %s", exc)
        return ErrorResult(error=str(exc))

    full_relevancy: List[RelevancyEntry] = []
    fm2sensor: Dict[str, List[str]] = {}
    sensor2fm: Dict[str, List[str]] = {}
    for batch, gen in zip(batches, generations):
        entry = RelevancyEntry(
            asset_name=batch["asset_name"],
            failure_mode=batch["failure_mode"],
            sensor=batch["sensor"],
            relevancy_answer=gen["answer"],
            relevancy_reason=gen["reason"],
            temporal_behavior=gen["temporal_behavior"],
        )
        full_relevancy.append(entry)
        if "yes" in gen["answer"].lower():
            fm2sensor.setdefault(batch["failure_mode"], []).append(batch["sensor"])
            sensor2fm.setdefault(batch["sensor"], []).append(batch["failure_mode"])

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
