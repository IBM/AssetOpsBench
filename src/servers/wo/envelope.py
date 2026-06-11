"""Response envelope helpers.

Identical shape to the Maximo MCP (`{success, data, metadata}` / `{success, error,
error_code}`) so AssetOpsBench agents written against the real Maximo server work
unchanged against this benchmark server.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


def envelope(data: Any, *, cached: bool = False, duration_ms: int = 0,
             record_count: Optional[int] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"cached": cached, "duration_ms": duration_ms}
    if record_count is not None:
        meta["record_count"] = record_count
    return {"success": True, "data": data, "metadata": meta}


def error(message: str, code: str = "API_ERROR") -> Dict[str, Any]:
    return {"success": False, "error": message, "error_code": code}


class Timer:
    """`with Timer() as t: ...; t.ms` — live millisecond wall time for the metadata block."""

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        pass

    @property
    def ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)