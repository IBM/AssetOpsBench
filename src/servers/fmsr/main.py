"""Failure-mode catalog reasoning for industrial asset classes."""

from __future__ import annotations

import difflib
import logging
import os
import re
from typing import List, Optional, Union

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Database stores ───────────────────────────────────────────────────────────
# Under AssetOpsBench's loader, database name == collection key. Failure modes
# live in the 'failure_mode' database; generic catalog lookups live in the
# utilities MCP server.

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
FAILURE_MODE_DBNAME = os.environ.get("FAILURE_MODE_DBNAME", "failure_mode")


def _connect(dbname):
    try:
        h = couchdb3.Database(
            dbname,
            url=COUCHDB_URL,
            user=COUCHDB_USERNAME,
            password=COUCHDB_PASSWORD,
        )
        logger.info("Connected to database: %s", dbname)
        return h
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to connect to database '%s': %s", dbname, e)
        return None


fm_db = _connect(FAILURE_MODE_DBNAME)


def _asset_class_key(asset_class: str) -> str:
    """Normalise an asset class for matching ('Hydraulic_Pump' -> 'hydraulic pump').

    Only case, punctuation, and whitespace are normalised; the caller is expected
    to pass a real class name, not an instance id like 'Pump-1'.
    """
    return re.sub(r"[\W_]+", " ", asset_class or "").strip().casefold()


# Upper bound on failure_mode records scanned per lookup. Matching is done in
# Python on normalised names, so every record's asset_class has to be read.
_MAX_ASSET_CLASSES = 10_000


def _load_failure_mode_docs(key: str) -> List[dict]:
    """Return every stored failure-mode doc that has an asset_class."""
    if not fm_db:
        raise RuntimeError("database not connected")
    try:
        res = fm_db.find({"asset_class": {"$exists": True}}, limit=_MAX_ASSET_CLASSES)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"database lookup failed for asset_class '{key}': {exc}"
        ) from exc
    return [
        doc
        for doc in res.get("docs", [])
        if isinstance(doc.get("asset_class"), str) and doc["asset_class"].strip()
    ]


def _match_failure_mode_doc(key: str, docs: List[dict]) -> Optional[dict]:
    """Return the doc whose normalised asset_class equals key, or None.

    Matching is on asset_class only, never on _id. If several docs normalise to
    the same key, the lowest _id wins.
    """
    matches = [doc for doc in docs if _asset_class_key(doc["asset_class"]) == key]
    return min(matches, key=lambda doc: str(doc.get("_id", "")), default=None)


def _missing_asset_class_error(
    original: str, normalized: str, docs: List[dict]
) -> ErrorResult:
    message = (
        f"no failure_mode record for asset_class '{normalized}' in database. "
        f"Input was normalized from {original!r}; check that asset_class matches a stored class."
    )
    by_key = {_asset_class_key(doc["asset_class"]): doc["asset_class"] for doc in docs}
    close = difflib.get_close_matches(normalized, list(by_key), n=3, cutoff=0.6)
    if close:
        message += f" Did you mean: {', '.join(by_key[k] for k in close)}?"
    known = sorted(set(by_key.values()))[:10]
    if known:
        message += f" Available asset_class values include: {', '.join(known)}."
    return ErrorResult(error=message)


# ── Result models ─────────────────────────────────────────────────────────────


class ErrorResult(BaseModel):
    error: str


class FailureModesResult(BaseModel):
    asset_class: str
    failure_modes: List[str]
    exhaustive: bool = False  # the stored list is not claimed to be complete
    source: Optional[str] = None  # provenance: ISO / curated / LLM:<model>


class AddFailureModesResult(BaseModel):
    asset_class: str
    added: List[str]
    failure_modes: List[str]
    total: int
    exhaustive: bool
    source: Optional[str] = None
    message: str


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "fmsr",
    instructions=(
        "Failure-mode catalog tools for industrial asset classes. Exposes stored "
        "failure-mode lookup and failure-mode persistence. This server does not "
        "generate failure modes or failure-mode/sensor mappings: when the stored "
        "list is missing or incomplete, use your own domain knowledge, and call "
        "add_failure_modes to persist any modes that should be kept."
    ),
)


@mcp.tool(title="Get Failure Modes")
def get_failure_modes(asset_class: str) -> Union[FailureModesResult, ErrorResult]:
    """READ the known failure modes for an asset class.

    Args:
        asset_class: Generic equipment type, such as "pump" or "hydraulic pump".
            Pass the class, not an asset id: for "Chiller 6" use "chiller". Case,
            punctuation, and whitespace are ignored. If no record is found, retry
            with a class suggested in the error.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    try:
        docs = _load_failure_mode_docs(key)
        d = _match_failure_mode_doc(key, docs)
        if d is None:
            return _missing_asset_class_error(asset_class, key, docs)
        return FailureModesResult(
            asset_class=d["asset_class"],
            failure_modes=d.get("failure_modes", []),
            exhaustive=d.get("exhaustive", False),
            source=d.get("source"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Add Failure Modes")
def add_failure_modes(
    asset_class: str,
    failure_modes: List[str],
    exhaustive: Optional[bool] = None,
    source: Optional[str] = None,
) -> Union[AddFailureModesResult, ErrorResult]:
    """WRITE failure modes for an asset class to the database.

    Existing modes are preserved, incoming modes are merged case-insensitively,
    and only newly added modes are reported. Use this when curated modes, or modes
    you derived from domain knowledge, should become available to future
    `get_failure_modes` calls.

    Args:
        asset_class: Generic equipment type, such as "pump", not an asset id.
            Reuse an existing class name where one fits, to avoid near-duplicate
            classes. Matched the same way as `get_failure_modes`; if nothing
            matches, a new record is created under the normalized name.
        failure_modes: Failure modes to add for the asset class.
        exhaustive: Set true only when the stored list is believed complete. If
            omitted, the existing value is preserved; new records default false.
        source: Optional provenance for the stored list.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    incoming = [
        mode.strip()
        for mode in (failure_modes or [])
        if isinstance(mode, str) and mode.strip()
    ]
    if not incoming:
        return ErrorResult(error="failure_modes list is required")
    if not fm_db:
        return ErrorResult(error="database not connected")

    try:
        doc = _match_failure_mode_doc(key, _load_failure_mode_docs(key))
        existing = [
            mode.strip()
            for mode in (doc or {}).get("failure_modes", [])
            if isinstance(mode, str) and mode.strip()
        ]
        seen = set()
        merged: List[str] = []
        for mode in existing:
            normalized = mode.lower()
            if normalized not in seen:
                seen.add(normalized)
                merged.append(mode)

        added: List[str] = []
        for mode in incoming:
            normalized = mode.lower()
            if normalized not in seen:
                seen.add(normalized)
                merged.append(mode)
                added.append(mode)

        if doc is None:
            doc = {"_id": f"fm:{key}", "asset_class": key}
            stored_exhaustive = False
        else:
            stored_exhaustive = bool(doc.get("exhaustive", False))
        doc["failure_modes"] = merged
        doc["exhaustive"] = stored_exhaustive if exhaustive is None else exhaustive
        doc["source"] = source or doc.get("source") or "user"
        fm_db.save(doc)

        return AddFailureModesResult(
            asset_class=doc["asset_class"],
            added=added,
            failure_modes=merged,
            total=len(merged),
            exhaustive=doc["exhaustive"],
            source=doc.get("source"),
            message=(
                f"added {len(added)} new failure mode(s) to asset_class '{doc['asset_class']}' "
                f"({len(merged)} total)."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("add_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


# Failure-mode generation and failure-mode/sensor mapping are intentionally not
# registered. Both are left to the agent's own model so benchmark results reflect
# the agent under test, not a separate server-side LLM.


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
