"""Optional pending-approval gate for destructive work-order actions.

Off by default — set ``AOB_REQUIRE_APPROVAL=1`` to turn it on (same opt-in
convention as ``AOB_READONLY``). When enabled, ``approve_workorder``,
``close_workorder``, and ``cancel_workorder`` do not execute on the first
call. That call is recorded as a *pending* action keyed by a SHA-256 digest
of the tool name and its arguments, and the tool returns an ``action_id``
instead of a result. An operator approves or denies the pending action out
of band (see ``scripts/wo_pending.py``); the agent must then repeat the
identical call. The matching digest consumes the approval exactly once and
the action runs. A retry with different arguments (a different `wonum`, a
different `reason`) is a new digest and a new pending decision — approving
one call never silently approves a different one.

If the pending store itself is unreachable, the gate fails closed (denies)
rather than letting the call through — the same posture ``AOB_READONLY``
already takes for reads.

This adds no new storage layer: pending records are plain CouchDB documents
in a dedicated database, read/written through the same ``CouchClient`` duck
type ``workorders.py`` already uses, so unit tests bind the same kind of
in-memory fake as ``test_workorders.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .couch import CouchClient, DatabaseUnavailable
from .envelope import error

PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"
CONSUMED = "consumed"
UNAVAILABLE = "unavailable"

REQUIRE_APPROVAL_ENV = "AOB_REQUIRE_APPROVAL"

# The three status-transition tools docs/mcp-servers.md documents as having
# no guard today. Scoped deliberately to these; read tools and additive
# writes (assign_technician, update_workorder) are left alone.
GATED_TOOLS = ("approve_workorder", "close_workorder", "cancel_workorder")


def approval_required() -> bool:
    return os.environ.get(REQUIRE_APPROVAL_ENV) == "1"


def args_digest(tool: str, wonum: str, site_id: str, args: Dict[str, Any]) -> str:
    """Stable SHA-256 over the tool name and every argument of the call.

    Same call, same digest. Change any argument and the digest changes, so an
    approval granted for one call can never cover a different one.
    """
    payload = json.dumps(
        {"tool": tool, "wonum": wonum, "site_id": site_id, "args": args},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _doc_id(action_id: str) -> str:
    return f"pending:{action_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingStore:
    """CouchDB-backed pending/approved/denied/consumed state for gated tools."""

    db: CouchClient

    async def request(
        self, tool: str, wonum: str, site_id: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create, or return the existing, pending record for this exact call."""
        digest = args_digest(tool, wonum, site_id, args)
        action_id = digest[:16]
        doc = await self.db.get(_doc_id(action_id))
        if doc is None:
            doc = {
                "_id": _doc_id(action_id),
                "action_id": action_id,
                "tool": tool,
                "wonum": wonum,
                "site_id": site_id,
                "args": args,
                "args_digest": digest,
                "status": PENDING,
                "created_at": _now(),
                "updated_at": _now(),
            }
            await self.db.put(doc)
        return doc

    async def decide(
        self, action_id: str, approve: bool, decided_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Operator approves or denies a pending action.

        Returns the updated record, or None if no such action is pending
        (already decided, consumed, or never existed).
        """
        doc = await self.db.get(_doc_id(action_id))
        if doc is None or doc.get("status") != PENDING:
            return None
        doc["status"] = APPROVED if approve else DENIED
        doc["decided_by"] = decided_by
        doc["updated_at"] = _now()
        await self.db.put(doc)
        return doc

    async def list_pending(self) -> list[Dict[str, Any]]:
        return await self.db.find({"status": PENDING})

    async def try_consume(
        self, tool: str, wonum: str, site_id: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check whether this exact call has a one-shot approval, and consume it.

        Returns ``{"ok": True, ...}`` when the caller may proceed, or
        ``{"ok": False, "action_id": ..., "status": ...}`` otherwise — including
        when the store itself is unreachable, which fails closed instead of
        letting the call through silently.
        """
        digest = args_digest(tool, wonum, site_id, args)
        action_id = digest[:16]
        try:
            doc = await self.db.get(_doc_id(action_id))
        except DatabaseUnavailable:
            return {"ok": False, "action_id": action_id, "status": UNAVAILABLE}

        if doc is None:
            created = await self.request(tool, wonum, site_id, args)
            return {"ok": False, "action_id": created["action_id"], "status": PENDING}

        if doc.get("status") != APPROVED:
            return {"ok": False, "action_id": action_id, "status": doc.get("status")}

        doc["status"] = CONSUMED
        doc["updated_at"] = _now()
        await self.db.put(doc)
        return {"ok": True, "action_id": action_id, "status": CONSUMED}


def pending_response(tool: str, action_id: str, status: str) -> Dict[str, Any]:
    """The ``{success: False, ...}`` envelope returned instead of executing."""
    if status == PENDING:
        msg = (
            f"'{tool}' requires operator approval before it can run "
            f"(action_id={action_id}). Ask an operator to review it with "
            "scripts/wo_pending.py, then repeat this exact call once approved."
        )
    elif status == DENIED:
        msg = f"'{tool}' was denied by an operator (action_id={action_id})."
    elif status == UNAVAILABLE:
        msg = (
            f"'{tool}' requires operator approval, but the approval store is "
            "unreachable; refusing to run rather than skipping the check."
        )
    else:
        msg = (
            f"'{tool}' is not approved to run (action_id={action_id}, status={status})."
        )
    return error(msg, "APPROVAL_REQUIRED")
