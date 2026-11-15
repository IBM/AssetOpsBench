"""Unit tests for the optional pending-approval gate (governance.py).

Uses the same in-memory fake pattern as test_workorders.py so no CouchDB is
needed. These tests cover the one-shot approval contract directly; the
request/response wiring through main.py's gated tools is exercised in
test_workorders.py's existing mutation tests with AOB_REQUIRE_APPROVAL unset
(default path, unaffected by this change).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from servers.wo import governance as gov
from servers.wo import main
from servers.wo import workorders as wo
from servers.wo.couch import DatabaseUnavailable
from servers.wo.models import ErrorResult, WorkOrderMutationResult


class FakeCouch:
    """Minimal get/put/find fake — the subset governance.py actually uses."""

    def __init__(self):
        self.docs = {}

    async def get(self, doc_id):
        d = self.docs.get(doc_id)
        return dict(d) if d else None

    async def put(self, doc):
        self.docs[doc["_id"]] = dict(doc)
        return {"id": doc["_id"], "rev": "1-x"}

    async def find(self, selector, **kw):
        return [
            dict(d)
            for d in self.docs.values()
            if all(d.get(k) == v for k, v in selector.items())
        ]


class UnavailableCouch:
    """A store that always raises, to exercise the fail-closed path."""

    async def get(self, doc_id):
        raise DatabaseUnavailable()

    async def put(self, doc):
        raise DatabaseUnavailable()


def test_approval_required_is_off_by_default(monkeypatch):
    monkeypatch.delenv(gov.REQUIRE_APPROVAL_ENV, raising=False)
    assert gov.approval_required() is False


def test_approval_required_honors_env(monkeypatch):
    monkeypatch.setenv(gov.REQUIRE_APPROVAL_ENV, "1")
    assert gov.approval_required() is True


def test_digest_changes_with_any_argument():
    base = gov.args_digest("cancel_workorder", "1000045", "MAIN", {"reason": "dup"})
    diff_wonum = gov.args_digest(
        "cancel_workorder", "1000046", "MAIN", {"reason": "dup"}
    )
    diff_reason = gov.args_digest(
        "cancel_workorder", "1000045", "MAIN", {"reason": "x"}
    )
    assert base != diff_wonum
    assert base != diff_reason


@pytest.mark.anyio
async def test_first_call_creates_pending_and_refuses():
    store = gov.PendingStore(FakeCouch())
    result = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    assert result["ok"] is False
    assert result["status"] == gov.PENDING
    assert len(result["action_id"]) == 16


@pytest.mark.anyio
async def test_repeat_call_before_decision_stays_pending_no_duplicate():
    db = FakeCouch()
    store = gov.PendingStore(db)
    first = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    second = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    assert first["action_id"] == second["action_id"]
    assert second["status"] == gov.PENDING
    assert len(db.docs) == 1  # one record, not re-created on every retry


@pytest.mark.anyio
async def test_approve_then_matching_retry_consumes_once():
    db = FakeCouch()
    store = gov.PendingStore(db)
    pending = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    decided = await store.decide(pending["action_id"], approve=True, decided_by="alice")
    assert decided["status"] == gov.APPROVED

    first_retry = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    assert first_retry["ok"] is True
    assert first_retry["status"] == gov.CONSUMED

    # Same call again: already consumed, not approved anymore -> refused again.
    second_retry = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    assert second_retry["ok"] is False
    assert second_retry["status"] == gov.CONSUMED


@pytest.mark.anyio
async def test_approval_does_not_cover_a_different_call():
    """Approving $500 must not approve a later call for $5,000,000."""
    db = FakeCouch()
    store = gov.PendingStore(db)
    pending = await store.try_consume(
        "close_workorder", "1000045", "MAIN", {"actual_hours": 2.0}
    )
    await store.decide(pending["action_id"], approve=True)

    different_call = await store.try_consume(
        "close_workorder", "1000045", "MAIN", {"actual_hours": 200.0}
    )
    assert different_call["ok"] is False
    assert different_call["status"] == gov.PENDING  # a brand new decision, not denied


@pytest.mark.anyio
async def test_deny_refuses_and_does_not_consume():
    db = FakeCouch()
    store = gov.PendingStore(db)
    pending = await store.try_consume(
        "cancel_workorder", "1000045", "MAIN", {"reason": "x"}
    )
    await store.decide(pending["action_id"], approve=False)

    retry = await store.try_consume(
        "cancel_workorder", "1000045", "MAIN", {"reason": "x"}
    )
    assert retry["ok"] is False
    assert retry["status"] == gov.DENIED


@pytest.mark.anyio
async def test_decide_on_unknown_action_returns_none():
    store = gov.PendingStore(FakeCouch())
    assert await store.decide("not-a-real-id", approve=True) is None


@pytest.mark.anyio
async def test_unreachable_store_fails_closed():
    store = gov.PendingStore(UnavailableCouch())
    result = await store.try_consume("approve_workorder", "1000045", "MAIN", {})
    assert result["ok"] is False
    assert result["status"] == gov.UNAVAILABLE


def test_pending_response_shapes_match_error_envelope():
    resp = gov.pending_response("approve_workorder", "abc123", gov.PENDING)
    assert resp["success"] is False
    assert resp["error_code"] == "APPROVAL_REQUIRED"
    assert "abc123" in resp["error"]


# --------------------------------------------------------------------------- #
# End-to-end through main.py's actual tools, not just the governance module
# in isolation — proves the gate really blocks and really unblocks a real
# approve_workorder call, and that it is a no-op when the env var is unset.
# --------------------------------------------------------------------------- #
T0 = datetime(2020, 4, 28, 9, 0, 0, tzinfo=timezone.utc)


async def _seeded_wo_db() -> FakeCouch:
    db = FakeCouch()
    await wo.create_workorder(
        db,
        description="Investigate Chiller 6 anomaly",
        asset_num="CHILLER6",
        site_id="MAIN",
        wonum="1000045",
        now=T0,
    )
    return db


@pytest.mark.anyio
async def test_gate_is_a_noop_when_env_var_unset(monkeypatch):
    monkeypatch.delenv(gov.REQUIRE_APPROVAL_ENV, raising=False)
    monkeypatch.setattr(main, "_db", await _seeded_wo_db())

    result = await main.approve_workorder("1000045", "MAIN")

    assert isinstance(result, WorkOrderMutationResult)
    assert result.status == "APPR"


@pytest.mark.anyio
async def test_gate_blocks_then_unblocks_a_real_approve_call(monkeypatch):
    monkeypatch.setenv(gov.REQUIRE_APPROVAL_ENV, "1")
    monkeypatch.setattr(main, "_db", await _seeded_wo_db())
    monkeypatch.setattr(main, "_pending_db", FakeCouch())

    first = await main.approve_workorder("1000045", "MAIN")
    assert isinstance(first, ErrorResult)
    assert "action_id" in first.error or "requires operator approval" in first.error

    # The work order must still be WAPPR — the gated call never ran wo.approve_workorder.
    still_open = await wo.get_workorder(main._db, "1000045", "MAIN")
    assert still_open["data"]["status"] == "WAPPR"

    store = gov.PendingStore(main._pending_db)
    [pending] = await store.list_pending()
    await store.decide(pending["action_id"], approve=True, decided_by="dhaval")

    second = await main.approve_workorder("1000045", "MAIN")
    assert isinstance(second, WorkOrderMutationResult)
    assert second.status == "APPR"
