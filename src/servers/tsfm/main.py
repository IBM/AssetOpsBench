"""TSFM feature catalog MCP server."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .core.store import make_store
from .models import (
    CardResult,
    ErrorResult,
    FeaturesResult,
    LineageResult,
    RegisterResult,
)
from .stores import feature_store

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")

mcp = FastMCP(
    "tsfm",
    instructions=(
        "Time-series feature catalog tools backed by CouchDB. Browse, search, "
        "register, update, deprecate, version, and inspect lineage for feature "
        "catalog cards."
    ),
)

_FEATURE_STORE = make_store()


def _validate_feature_kind(kind: Optional[str]) -> Optional[ErrorResult]:
    if kind is not None and kind not in {"extractor", "transform"}:
        return ErrorResult(error="kind must be 'extractor', 'transform', or omitted")
    return None


@mcp.tool(title="List Feature Catalog")
def list_features(
    kind: Optional[str] = None,
    status: Optional[str] = "active",
) -> Union[FeaturesResult, ErrorResult]:
    """List feature catalog cards from CouchDB.

    Args:
        kind: Optional feature kind filter: "transform" or "extractor".
        status: Optional status filter. Defaults to active; pass null or "" for all.
    """
    err = _validate_feature_kind(kind)
    if err:
        return err
    try:
        return FeaturesResult(
            features=feature_store.find_features(
                _FEATURE_STORE, kind=kind, status=status
            )
        )
    except Exception as exc:
        logger.error("list_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Search Feature Catalog")
def search_features(
    text: str = "",
    tags: Optional[List[str]] = None,
    status: Optional[str] = "active",
) -> Union[FeaturesResult, ErrorResult]:
    """Search feature catalog cards by id, name, description, or tags."""
    try:
        return FeaturesResult(
            features=feature_store.search(
                _FEATURE_STORE, text=text, tags=tags, status=status
            )
        )
    except Exception as exc:
        logger.error("search_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Feature")
def get_feature(feature_id: str) -> Union[CardResult, ErrorResult]:
    """Return one feature catalog card by feature_id."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        card = feature_store.get_feature(_FEATURE_STORE, feature_id)
        if not card:
            return ErrorResult(error=f"feature '{feature_id}' not found")
        return CardResult(**card)
    except Exception as exc:
        logger.error("get_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Register Feature")
def register_feature(
    feature: dict,
    overwrite: bool = False,
) -> Union[RegisterResult, ErrorResult]:
    """Register a transform feature card after schema and execution validation."""
    if not feature:
        return ErrorResult(error="feature card is required")
    try:
        rec = feature_store.register_feature(
            _FEATURE_STORE, feature, overwrite=overwrite
        )
        return RegisterResult(
            status="registered", id=rec.get("feature_id", ""), card=rec
        )
    except Exception as exc:
        logger.error("register_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Update Feature")
def update_feature(feature_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on a feature catalog card."""
    if not feature_id.strip() or not fields:
        return ErrorResult(error="feature_id and fields are required")
    try:
        return CardResult(
            **feature_store.update_feature(_FEATURE_STORE, feature_id, fields)
        )
    except Exception as exc:
        logger.error("update_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Deprecate Feature")
def deprecate_feature(
    feature_id: str, reason: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Mark a feature catalog card as deprecated."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return CardResult(
            **feature_store.deprecate_feature(
                _FEATURE_STORE, feature_id, reason=reason
            )
        )
    except Exception as exc:
        logger.error("deprecate_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="New Feature Version")
def new_feature_version(
    feature_id: str,
    fields: Optional[dict] = None,
    new_feature_id: Optional[str] = None,
) -> Union[CardResult, ErrorResult]:
    """Create a successor version for a transform feature and supersede the predecessor."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return CardResult(
            **feature_store.new_version(
                _FEATURE_STORE,
                feature_id,
                fields or {},
                new_feature_id=new_feature_id,
            )
        )
    except Exception as exc:
        logger.error("new_feature_version failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Feature Lineage")
def get_feature_lineage(feature_id: str) -> Union[LineageResult, ErrorResult]:
    """Return the parent and descendant chain for a feature catalog card."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return LineageResult(**feature_store.get_lineage(_FEATURE_STORE, feature_id))
    except Exception as exc:
        logger.error("get_feature_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
