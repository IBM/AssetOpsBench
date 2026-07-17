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
    """List feature catalog cards from the configured CouchDB database.

    Use this to browse candidate transform or extractor cards before choosing a
    feature for a workflow. The runtime database name comes from
    `FEATURE_CATALOG_DBNAME`, defaulting to `feature_catalog`.

    Args:
        kind: Optional exact feature kind filter. Use `transform` for executable
            fit/transform programs, `extractor` for scalar extractor metadata, or
            omit for both. Any other value returns ErrorResult.
        status: Optional exact status filter. Defaults to `active`; pass null or
            an empty string to include deprecated and superseded cards.

    Returns:
        FeaturesResult: Matching feature cards as stored in CouchDB. Each card
        includes fields such as `feature_id`, `kind`, `status`, `description`,
        and any card-specific metadata.
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
    """Search feature catalog cards by id, name, description, or tags.

    The search is a case-insensitive literal substring match, not semantic
    retrieval. Use `list_features()` when you need a complete catalog browse.

    Args:
        text: Optional substring to match against `feature_id`, `name`,
            `description`, and tags. Empty string returns all cards allowed by
            the status and tag filters.
        tags: Optional list of tags that must all be present on a card.
        status: Optional exact status filter. Defaults to `active`; pass null or
            an empty string to search every status.

    Returns:
        FeaturesResult: Matching feature cards. The result can be empty when no
        cards satisfy the filters.
    """
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
    """Return one feature catalog card by feature id.

    Use this after `list_features()` or `search_features()` when the full card
    is needed, including executable transform code and validity metadata.

    Args:
        feature_id: Exact feature id without the CouchDB `feature:` prefix, such
            as `efe_time_robust_norm_v1`. Empty input returns ErrorResult.

    Returns:
        CardResult: The stored feature card with CouchDB revision metadata
        stripped. Returns ErrorResult when the id is blank, absent, or the
        backing database query fails.
    """
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
    """Register an executable transform feature card.

    Registration is for `kind=transform` cards only. The card is validated
    against the feature schema and its `code` is executed through the feature
    runner to verify required entry points, no in-place mutation, and optional
    invertibility.

    Args:
        feature: Feature card payload. Required fields are `feature_id`,
            `interface`, and `code`; optional fields include `name`,
            `description`, `target_task`, `target_model`, `tags`, and
            `invertible`.
        overwrite: When false, an existing `feature_id` returns ErrorResult.
            When true, the card replaces the existing document.

    Returns:
        RegisterResult: Registration status, feature id, and the stored card.
        Returns ErrorResult for missing payload, schema errors, failed execution
        validation, duplicate ids, or CouchDB write failures.
    """
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
    """Patch fields on an existing feature catalog card.

    This is a direct metadata patch for catalog maintenance. It does not rerun
    transform-code validation, so use `register_feature()` or
    `new_feature_version()` when changing executable transform code.

    Args:
        feature_id: Exact feature id without the CouchDB `feature:` prefix.
            Empty input returns ErrorResult.
        fields: Non-empty mapping of fields to merge into the stored card.

    Returns:
        CardResult: The updated feature card, including `updated_at`. Returns
        ErrorResult when inputs are blank, the feature does not exist, or the
        backing database write fails.
    """
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
    """Mark a feature catalog card as deprecated.

    Deprecation keeps the document available for lineage and audit purposes but
    removes it from default `active` list/search results.

    Args:
        feature_id: Exact feature id without the CouchDB `feature:` prefix.
            Empty input returns ErrorResult.
        reason: Optional human-readable reason stored as `deprecation_reason`.

    Returns:
        CardResult: The updated feature card with `status=deprecated`. Returns
        ErrorResult when the id is blank, unknown, or the backing database write
        fails.
    """
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
    """Create a successor version for a transform feature.

    Only `kind=transform` cards can be versioned. The new card is validated like
    `register_feature()`, receives a bumped `version`, points back through
    `parent_feature_id`, and the predecessor is marked `superseded`.

    Args:
        feature_id: Exact transform feature id to version. Empty input returns
            ErrorResult.
        fields: Optional mapping of changes to apply to the successor card.
        new_feature_id: Optional explicit id for the successor. When omitted,
            the id is generated as `<feature_id>_v<version>`.

    Returns:
        CardResult: The newly stored successor feature card. Returns ErrorResult
        for blank or unknown ids, extractor cards, validation failures,
        duplicate successor ids, or CouchDB write failures.
    """
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
    """Return the parent and descendant chain for a feature catalog card.

    Lineage is most useful for transform cards created through
    `new_feature_version()`. Extractor cards can be queried but typically return
    an empty ancestor and descendant list.

    Args:
        feature_id: Exact feature id without the CouchDB `feature:` prefix.
            Empty input returns ErrorResult.

    Returns:
        LineageResult: `feature_id`, ordered ancestor ids, root id, and direct
        descendant ids. Returns ErrorResult when the id is blank or the backing
        database query fails.
    """
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
