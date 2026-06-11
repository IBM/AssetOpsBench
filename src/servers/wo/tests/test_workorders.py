"""Smoke tests for the WO tools using an in-memory fake CouchDB (no server needed).

Run:  python -m pytest tests/ -q     (or just `python tests/test_workorders.py`)
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from servers.wo import workorders as wo


class FakeCouch:
    """Implements the small CouchClient interface against a dict, including the
    subset of Mango operators the tools use ($in, $gte, $lte, $elemMatch)."""

    def __init__(self):
        self.docs = {}
        self._rev = 0

    async def get(self, doc_id):
        d = self.docs.get(doc_id)
        return dict(d) if d else None

    async def put(self, doc):
        self._rev += 1
        doc = dict(doc)
        doc["_rev"] = f"{self._rev}-x"
        self.docs[doc["_id"]] = doc
        return {"id": doc["_id"], "rev": doc["_rev"]}

    async def delete(self, doc_id, rev):
        self.docs.pop(doc_id, None)
        return {"ok": True}

    async def next_wonum(self, site_id):
        cid = f"counter:{site_id.upper()}"
        doc = self.docs.get(cid) or {"_id": cid, "type": "counter", "value": 1000}
        doc = dict(doc)
        doc["value"] = int(doc["value"]) + 1
        await self.put(doc)
        return str(doc["value"])

    def _match(self, doc, selector):
        for k, cond in selector.items():
            val = doc.get(k)
            if isinstance(cond, dict):
                for op, arg in cond.items():
                    if op == "$in" and val not in arg:
                        return False
                    if op == "$nin" and val in arg:
                        return False
                    if op == "$gte" and not (val is not None and val >= arg):
                        return False
                    if op == "$lte" and not (val is not None and val <= arg):
                        return False
                    if op == "$lt" and not (val is not None and val < arg):
                        return False
                    if op == "$elemMatch":
                        if not isinstance(val, list) or not any(
                            all(it.get(ik) == iv for ik, iv in arg.items()) for it in val):
                            return False
            else:
                if val != cond:
                    return False
        return True

    async def find(self, selector, fields=None, sort=None, limit=200, skip=0):
        rows = [dict(d) for d in self.docs.values() if self._match(d, selector)]
        if sort:
            key = list(sort[0].keys())[0]
            rev = sort[0][key] == "desc"
            rows.sort(key=lambda d: d.get(key) or "", reverse=rev)
        return rows[skip:skip + limit]


T0 = datetime(2020, 4, 28, 9, 0, 0, tzinfo=timezone.utc)


async def scenario():
    db = FakeCouch()

    # create with provenance + pinned wonum for reproducibility
    r = await wo.create_workorder(
        db, description="Investigate Chiller 6 anomaly", asset_num="CHILLER6",
        site_id="MAIN", priority=2, work_type="PdM", reported_by="AGENT.TSFM",
        wonum="1000045", now=T0,
        aob_source={"agent": "tsfm", "trigger_type": "anomaly_detection",
                    "scenario_id": "WO-CHILLER6-ANOMALY-001"})
    assert r["success"], r
    assert r["data"]["_id"] == "wo:MAIN:1000045"
    assert r["data"]["status"] == "WAPPR"
    assert "_rev" not in r["data"], "internal _rev must not leak"
    assert r["data"]["reportdate"] == "2020-04-28T09:00:00+00:00", "clock must be injectable"

    # get
    g = await wo.get_workorder(db, "1000045", "MAIN")
    assert g["success"] and g["data"]["aob_source"]["agent"] == "tsfm"

    # not found
    nf = await wo.get_workorder(db, "999", "MAIN")
    assert not nf["success"] and nf["error_code"] == "NOT_FOUND"

    # approve -> assign -> close
    assert (await wo.approve_workorder(db, "1000045", "MAIN", now=T0))["data"]["status"] == "APPR"
    a = await wo.assign_technician(db, "1000045", "MAIN", "HVACTECH1", craft="HVAC",
                                   hours_planned=4, now=T0)
    assert a["data"]["wplabor"][0]["laborcode"] == "HVACTECH1"
    c = await wo.close_workorder(db, "1000045", "MAIN", actual_hours=3.5,
                                 failure_code="SENSOR-DRIFT", now=T0)
    assert c["data"]["status"] == "COMP" and c["data"]["actlabhrs"] == 3.5
    assert c["data"]["actfinish"] == "2020-04-28T09:00:00+00:00"

    # auto wonum allocation is sequential/deterministic after reset
    w1 = (await wo.create_workorder(db, description="PM", asset_num="AHU2", site_id="MAIN",
                                    work_type="PM", now=T0))["data"]["wonum"]
    w2 = (await wo.create_workorder(db, description="PM", asset_num="AHU3", site_id="MAIN",
                                    work_type="PM", now=T0))["data"]["wonum"]
    assert int(w2) == int(w1) + 1, (w1, w2)

    # list + filters
    lst = await wo.list_workorders(db, site_id="MAIN", status="OPEN")
    open_nums = {w["wonum"] for w in lst["data"]["workorders"]}
    assert "1000045" not in open_nums  # it's COMP now
    assert w1 in open_nums and w2 in open_nums

    # validation errors
    bad = await wo.create_workorder(db, description="x", asset_num="A", site_id="S", priority=9)
    assert not bad["success"] and bad["error_code"] == "VALIDATION_ERROR"

    # my assigned (open_only excludes the closed one)
    mine = await wo.get_my_assigned_workorders(db, "HVACTECH1", site_id="MAIN", open_only=True)
    assert mine["data"]["totalCount"] == 0

    # kpis
    k = await wo.get_workorder_kpis(db, "MAIN", period_months=3,
                                    now=datetime(2020, 5, 1, tzinfo=timezone.utc))
    assert k["data"]["completed"] == 1 and k["data"]["total_workorders"] >= 3

    print("ALL ASSERTIONS PASSED")


def test_lifecycle():
    asyncio.run(scenario())


if __name__ == "__main__":
    asyncio.run(scenario())
