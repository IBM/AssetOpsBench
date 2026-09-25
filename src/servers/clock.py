"""Shared UTC clock for the MCP servers.

Every server reads "now" through :func:`now_utc` instead of calling
``datetime.now`` directly.  Setting ``ASSETOPS_FIXED_DATETIME`` to an ISO-8601
timestamp freezes that clock across all servers, which makes benchmark runs
reproducible: questions anchored to "today", work-order KPI windows, and
telemetry-recency checks then resolve identically on every run.

The variable is read on each call (not at import time) so it can come from
``.env`` — which every server loads — or be monkeypatched in tests.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

FIXED_DATETIME_ENV = "ASSETOPS_FIXED_DATETIME"


def fixed_datetime() -> datetime | None:
    """Return the pinned clock from the environment, or ``None`` if unset.

    Raises:
        ValueError: If ``ASSETOPS_FIXED_DATETIME`` is set but unparseable.
    """
    raw = os.environ.get(FIXED_DATETIME_ENV, "").strip()
    if not raw:
        return None

    try:
        pinned = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{FIXED_DATETIME_ENV}={raw!r} is not a valid ISO-8601 datetime "
            "(expected e.g. '2025-01-15T09:00:00Z')"
        ) from exc

    if pinned.tzinfo is None:
        pinned = pinned.replace(tzinfo=timezone.utc)
    return pinned.astimezone(timezone.utc)


def now_utc() -> datetime:
    """Return the current UTC time, or the pinned one when the env var is set."""
    return fixed_datetime() or datetime.now(timezone.utc)
