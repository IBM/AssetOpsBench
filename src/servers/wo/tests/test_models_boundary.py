"""Tests for the Pydantic output boundary in main.py, including partial-column docs.

Injects an in-memory fake CouchDB into main._db, so no server/CouchDB is needed.
Run:  PYTHONPATH=.. python tests/test_models_boundary.py
"""
import asyncio
from datetime import datetime, timezone

from servers.wo.tests.test_workorders import FakeCouch  # reuse the in-memory fake
from servers.wo import main
from servers.wo.models import (
    ErrorResult, WorkOrderItem, WorkOrderMutationResult, WorkOrderResult, WorkOrdersResult,
)

T0 = datetime(2020, 4, 28, 9, 0, 0, tzinfo=timezone.utc)


async def scenario():
    db = FakeCouch()
    main._db = db  # inject fake

    # 1) Output is a Pydantic model, not a dict
    created = await main.generate_work_order(description="Chiller 6 anomaly", asset_num="CHILLER6",
                                             site_id="MAIN", priority=2, work_type="PdM", wonum="1000045")
    assert isinstance(created, WorkOrderMutationResult), type(created)
    assert created.status == "WAPPR" and created.work_order.assetnum == "CHILLER6"

    got = await main.get_workorder("1000045", "MAIN")
    assert isinstance(got, WorkOrderResult)
    assert got.work_order.wonum == "1000045"

    listed = await main.list_workorders(site_id="MAIN")
    assert isinstance(listed, WorkOrdersResult) and listed.total == 1

    # 2) Errors come back as ErrorResult (typed), not an exception
    missing = await main.get_workorder("999", "MAIN")
    assert isinstance(missing, ErrorResult) and "not found" in missing.error.lower()

    # 3) THE KEY CASE: a work order missing most columns still converts cleanly
    partial = {"_id": "wo:MAIN:2000", "type": "workorder", "wonum": "2000",
               "siteid": "MAIN", "status": "APPR", "worktype": "CM",
               "description": "partial doc", "reportdate": "2020-01-01T00:00:00+00:00"}
    # store it directly and read back through the typed boundary
    await db.put(partial)
    res = await main.get_workorder("2000", "MAIN")
    assert isinstance(res, WorkOrderResult)
    wo = res.work_order
    assert wo.wonum == "2000"
    # every absent column is simply None — no crash, no missing-key error
    assert wo.assetnum is None and wo.wopriority is None and wo.actlabhrs is None
    assert wo.wplabor is None and wo.failurecode is None

    # 4) extra/unmodeled fields are preserved (extra='allow')
    extra_doc = {"_id": "wo:MAIN:2001", "type": "workorder", "wonum": "2001", "siteid": "MAIN",
                 "status": "APPR", "worktype": "CM", "description": "x",
                 "reportdate": "2020-01-01T00:00:00+00:00", "custom_field": "kept"}
    await db.put(extra_doc)
    r2 = await main.get_workorder("2001", "MAIN")
    assert r2.work_order.model_dump().get("custom_field") == "kept"

    print("ALL ASSERTIONS PASSED")


def test_boundary():
    asyncio.run(scenario())


if __name__ == "__main__":
    asyncio.run(scenario())
