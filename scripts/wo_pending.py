#!/usr/bin/env python3
"""Operator CLI for the WO server's pending-approval gate (AOB_REQUIRE_APPROVAL).

Lists, approves, or denies pending approve_workorder / close_workorder /
cancel_workorder calls. Talks directly to CouchDB using the same
COUCHDB_URL / COUCHDB_USERNAME / COUCHDB_PASSWORD / WO_PENDING_DBNAME env vars
the WO server itself uses — see src/servers/wo/governance.py.

Usage:
    wo_pending.py list
    wo_pending.py approve <action_id> [--by NAME]
    wo_pending.py deny <action_id> [--by NAME]

Exit code 0 = command completed, 1 = action not found / nothing to do,
2 = the command itself failed (bad args, store unreachable).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from servers.wo.couch import CouchClient, DatabaseUnavailable
from servers.wo.governance import PendingStore


def _pending_db() -> CouchClient:
    return CouchClient(
        base_url=os.environ.get("COUCHDB_URL", "http://localhost:5984"),
        db=os.environ.get("WO_PENDING_DBNAME", "workorder_pending"),
        username=os.environ.get("COUCHDB_USERNAME", "admin"),
        password=os.environ.get("COUCHDB_PASSWORD", "password"),
    )


async def _list(store: PendingStore) -> int:
    rows = await store.list_pending()
    if not rows:
        print("No pending actions.")
        return 0
    for row in rows:
        print(
            f"{row['action_id']}  {row['tool']:<18} wonum={row['wonum']} "
            f"site={row['site_id']} args={row.get('args')} "
            f"created={row['created_at']}"
        )
    return 0


async def _decide(
    store: PendingStore, action_id: str, approve: bool, by: str | None
) -> int:
    doc = await store.decide(action_id, approve=approve, decided_by=by)
    if doc is None:
        print(f"No pending action '{action_id}' (already decided, or no such id).")
        return 1
    verb = "approved" if approve else "denied"
    print(
        f"{verb}: {action_id} ({doc['tool']} on wonum={doc['wonum']} site={doc['site_id']})"
    )
    if approve:
        print("The agent must repeat the identical call once to consume this approval.")
    return 0


async def _run(args: argparse.Namespace) -> int:
    db = _pending_db()
    store = PendingStore(db)
    try:
        if args.command == "list":
            return await _list(store)
        if args.command == "approve":
            return await _decide(store, args.action_id, approve=True, by=args.by)
        if args.command == "deny":
            return await _decide(store, args.action_id, approve=False, by=args.by)
    except DatabaseUnavailable:
        print("Pending-action store is unreachable (CouchDB down or misconfigured).")
        return 2
    finally:
        await db.aclose()
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all pending actions.")

    approve = sub.add_parser("approve", help="Approve a pending action.")
    approve.add_argument("action_id")
    approve.add_argument("--by", default=None, help="Operator identity to record.")

    deny = sub.add_parser("deny", help="Deny a pending action.")
    deny.add_argument("action_id")
    deny.add_argument("--by", default=None, help="Operator identity to record.")

    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
