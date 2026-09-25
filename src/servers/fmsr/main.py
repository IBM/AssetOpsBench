"""Failure-mode catalog reasoning for industrial asset classes."""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Union

import couchdb3
from couchdb3.exceptions import NotFoundError
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
    """Normalise an asset class to the database key format ('Hydraulic_Pump' -> 'hydraulic pump')."""
    key = re.sub(r"\d+", "", asset_class or "")
    key = re.sub(r"[_\-]+", " ", key)
    return re.sub(r"\s+", " ", key).strip().lower()


def _known_asset_classes(limit: int = 10) -> List[str]:
    """Return known asset classes from the failure_mode collection for error guidance."""
    if not fm_db:
        return []
    try:
        res = fm_db.find({}, fields=["asset_class"], limit=limit)
    except Exception:  # noqa: BLE001
        return []
    classes = [
        doc.get("asset_class")
        for doc in res.get("docs", [])
        if isinstance(doc.get("asset_class"), str) and doc.get("asset_class")
    ]
    return sorted(dict.fromkeys(classes))


def _missing_asset_class_error(original: str, normalized: str) -> ErrorResult:
    message = (
        f"no failure_mode record for asset_class '{normalized}' in database. "
        f"Input was normalized from {original!r}; check that asset_class matches a stored class."
    )
    known = _known_asset_classes()
    if known:
        message += f" Available asset_class values include: {', '.join(known)}."
    return ErrorResult(error=message)


def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, NotFoundError)):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        exc, "status_code", None
    )
    return status_code == 404


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
        asset_class: Asset class to look up, such as "pump". Case, whitespace,
            digits, underscores, and hyphens are normalized before querying.
    """
    raw_asset_class = asset_class
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    try:
        d = _find_failure_mode_doc(key)
        if d is None:
            return _missing_asset_class_error(raw_asset_class, key)
        return FailureModesResult(
            asset_class=d.get("asset_class", key),
            failure_modes=d.get("failure_modes", []),
            exhaustive=d.get("exhaustive", False),
            source=d.get("source"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


def _find_failure_mode_doc(asset_class: str) -> Optional[dict]:
    """Return the stored failure-mode doc for an asset class, or None."""
    if not fm_db:
        raise RuntimeError("database not connected")
    key = _asset_class_key(asset_class)
    try:
        d = fm_db.get(f"fm:{key}", check=True)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found_error(exc):
            d = None
        else:
            raise RuntimeError(
                f"database lookup failed for asset_class '{key}': {exc}"
            ) from exc
    try:
        if d is None:
            res = fm_db.find({"asset_class": key}, limit=1)
            docs = res["docs"]
            if docs:
                d = docs[0]
        return d
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"database lookup failed for asset_class '{key}': {exc}"
        ) from exc


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
        asset_class: Asset class to update, such as "pump". Case, whitespace,
            digits, underscores, and hyphens are normalized before writing.
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
        doc_id = f"fm:{key}"
        doc = _find_failure_mode_doc(key)
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
            doc = {"_id": doc_id, "asset_class": key}
            stored_exhaustive = False
        else:
            doc.setdefault("_id", doc_id)
            doc["asset_class"] = key
            stored_exhaustive = bool(doc.get("exhaustive", False))
        doc["failure_modes"] = merged
        doc["exhaustive"] = stored_exhaustive if exhaustive is None else exhaustive
        doc["source"] = source or doc.get("source") or "user"
        fm_db.save(doc)

        return AddFailureModesResult(
            asset_class=key,
            added=added,
            failure_modes=merged,
            total=len(merged),
            exhaustive=doc["exhaustive"],
            source=doc.get("source"),
            message=(
                f"added {len(added)} new failure mode(s) to asset_class '{key}' "
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
