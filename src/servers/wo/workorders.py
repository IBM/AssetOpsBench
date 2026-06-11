"""Work-order lifecycle tools for AssetOpsBench, backed by CouchDB.

Mirrors the tool surface of the Maximo MCP `tools/workorders.py`, but instead of
issuing OSLC calls to a live Maximo, every operation reads/writes the benchmark
CouchDB. Field names are identical to Maximo `mxwo` (see workorder.schema.json).

Design choices that differ from the real Maximo MCP (and why):
  * Deterministic `_id` (`wo:{SITEID}:{WONUM}`) → direct GET, no wonum-only fetch +
    Python post-filter, and no caching layer (the DB is local and authoritative).
  * Injected clock (`now`) so created timestamps are reproducible during grading.
  * No auth/RBAC transport errors to emulate; an optional `role` gate is kept so
    write tools can be disabled in read-only scenarios.

Every function takes the CouchDB client `db` as its first argument. `server.py`
binds a real client; tests bind an in-memory fake. Each returns the same
`{success, data, metadata}` / `{success, error, error_code}` envelope as Maximo MCP.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .envelope import envelope, error, Timer

OPEN_STATUSES = ("WAPPR", "APPR", "WMATL", "WSCH", "INPRG", "WPCOND")
APPROVED_PENDING = ("APPR", "WMATL", "WSCH", "INPRG", "WPCOND")
TERMINAL = ("COMP", "CLOSE", "CAN")
ALL_STATUSES = OPEN_STATUSES + ("COMP", "CLOSE", "CAN")
WORKTYPES = ("CM", "PM", "EM", "PdM", "CAL", "INSP", "GEN")


def _doc_id(site_id: str, wonum: str) -> str:
    return f"wo:{site_id.upper()}:{wonum}"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _public(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip CouchDB bookkeeping that agents shouldn't see."""
    return {k: v for k, v in doc.items() if k not in ("_rev",)}


# --------------------------------------------------------------------------- #
# Read tools
# --------------------------------------------------------------------------- #
async def list_workorders(db, site_id: Optional[str] = None, status: Optional[str] = None,
                          asset_num: Optional[str] = None, priority: Optional[int] = None,
                          date_from: Optional[str] = None, date_to: Optional[str] = None,
                          page_size: int = 50, page_num: int = 1) -> Dict[str, Any]:
    """List work orders with optional filters (site, status, asset, priority, date window).

    `status` accepts a single value, or the pseudo-values OPEN / APPROVED_PENDING.
    `page_size=0` returns ALL matching work orders in a single response (no paging) —
    fine here because the benchmark DB is local and bounded.
    """
    with Timer() as t:
        sel: Dict[str, Any] = {"type": "workorder"}
        if site_id:
            sel["siteid"] = site_id.upper()
        if asset_num:
            sel["assetnum"] = asset_num
        if priority is not None:
            sel["wopriority"] = priority
        if status:
            su = status.strip().upper()
            if su == "OPEN":
                sel["status"] = {"$in": list(OPEN_STATUSES)}
            elif su == "APPROVED_PENDING":
                sel["status"] = {"$in": list(APPROVED_PENDING)}
            else:
                sel["status"] = su
        if date_from or date_to:
            rng: Dict[str, Any] = {}
            if date_from:
                rng["$gte"] = date_from
            if date_to:
                rng["$lte"] = date_to
            sel["reportdate"] = rng

        # No Mango `sort` — that requires a matching index and 400s without one.
        # Sort client-side instead (robust to missing reportdate / indexes).
        docs = await db.find(sel, limit=1000000)
        docs.sort(key=lambda d: (d.get("reportdate") or ""), reverse=True)
        total = len(docs)
        if not page_size:  # page_size=0 (or None) → return everything
            page = [_public(d) for d in docs]
        else:
            start = (page_num - 1) * page_size
            page = [_public(d) for d in docs[start:start + page_size]]
        return envelope({"workorders": page, "totalCount": total},
                        duration_ms=t_ms(t), record_count=len(page))


async def get_workorder(db, wonum: str, site_id: str) -> Dict[str, Any]:
    """Get the full work-order document by number + site."""
    with Timer() as t:
        doc = await db.get(_doc_id(site_id, wonum))
        if not doc:
            return error(f"Work order '{wonum}' not found in site '{site_id}'", "NOT_FOUND")
        return envelope(_public(doc), duration_ms=t_ms(t))


async def get_workorder_tasks(db, wonum: str, site_id: str) -> Dict[str, Any]:
    """List child task rows whose `parent` references this work order."""
    with Timer() as t:
        docs = await db.find({"type": "workorder", "parent": wonum, "siteid": site_id.upper()},
                             limit=1000)
        docs.sort(key=lambda d: (d.get("taskid") or 0))   # sort client-side (no index needed)
        return envelope({"parent_wonum": wonum, "site_id": site_id,
                         "tasks": [_public(d) for d in docs]},
                        duration_ms=t_ms(t), record_count=len(docs))


async def get_workorder_costs(db, wonum: str, site_id: str) -> Dict[str, Any]:
    """Labor + material + service + tool actual-cost breakdown for one work order."""
    with Timer() as t:
        wo = await db.get(_doc_id(site_id, wonum))
        if not wo:
            return error(f"Work order '{wonum}' not found in site '{site_id}'.", "NOT_FOUND")
        f = lambda n: float(wo.get(n) or 0)
        labor, material, service, tool = f("actlabcost"), f("actmatcost"), f("actservcost"), f("acttoolcost")
        total = f("acttotalcost") or (labor + material + service + tool)
        breakdown = [{"category": c, "amount": round(a, 2),
                      "share_pct": round((a / total) * 100, 1) if total else 0}
                     for c, a in (("labor", labor), ("material", material),
                                  ("service", service), ("tool", tool))]
        return envelope({"wonum": wonum, "site_id": site_id, "status": wo.get("status"),
                         "assetnum": wo.get("assetnum"), "location": wo.get("location"),
                         "actual_hours": f("actlabhrs"), "total_cost": round(total, 2),
                         "breakdown": breakdown}, duration_ms=t_ms(t))


async def get_workorder_actuals_vs_planned(db, wonum: str, site_id: str) -> Dict[str, Any]:
    """Estimated vs actual hours and cost variance for one work order."""
    with Timer() as t:
        wo = await db.get(_doc_id(site_id, wonum))
        if not wo:
            return error(f"Work order '{wonum}' not found in site '{site_id}'.", "NOT_FOUND")
        f = lambda n: float(wo.get(n) or 0)

        def var(est, act):
            return {"estimated": round(est, 2), "actual": round(act, 2),
                    "variance_abs": round(act - est, 2),
                    "variance_pct": round(((act - est) / est) * 100, 1) if est else None,
                    "over_budget": act > est}

        est_total = f("esttotalcost") or (f("estlabcost") + f("estmatcost") + f("estservcost") + f("esttoolcost"))
        act_total = f("acttotalcost") or (f("actlabcost") + f("actmatcost") + f("actservcost") + f("acttoolcost"))
        return envelope({"wonum": wonum, "site_id": site_id, "status": wo.get("status"),
                         "worktype": wo.get("worktype"),
                         "labor_hours": var(f("estlabhrs"), f("actlabhrs")),
                         "labor_cost": var(f("estlabcost"), f("actlabcost")),
                         "material_cost": var(f("estmatcost"), f("actmatcost")),
                         "service_cost": var(f("estservcost"), f("actservcost")),
                         "tool_cost": var(f("esttoolcost"), f("acttoolcost")),
                         "total_cost": var(est_total, act_total)}, duration_ms=t_ms(t))


async def get_workorder_kpis(db, site_id: str, period_months: int = 3,
                             now: Optional[datetime] = None) -> Dict[str, Any]:
    """Site KPIs over a period: totals, backlog, overdue, avg completion, priority + asset breakdowns."""
    with Timer() as t:
        now = now or datetime.now(timezone.utc)
        cutoff = _iso(now - timedelta(days=period_months * 30))
        now_str = _iso(now)
        docs = await db.find({"type": "workorder", "siteid": site_id.upper()}, limit=10000)
        wos = [w for w in docs if (w.get("reportdate") or "") >= cutoff]
        completed = [w for w in wos if w.get("status") == "COMP"]
        backlog = [w for w in wos if w.get("status") not in TERMINAL]
        overdue = [w for w in backlog if w.get("targcompdate") and w["targcompdate"] < now_str]

        times = []
        for w in completed:
            try:
                s = datetime.fromisoformat(w["reportdate"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(w["actfinish"].replace("Z", "+00:00"))
                times.append((e - s).total_seconds() / 3600)
            except Exception:
                pass
        avg_hrs = round(sum(times) / len(times), 2) if times else 0

        prio: Dict[str, int] = {}
        assets: Dict[str, int] = {}
        for w in wos:
            prio[str(w.get("wopriority", "Unknown"))] = prio.get(str(w.get("wopriority", "Unknown")), 0) + 1
            a = w.get("assetnum", "UNKNOWN")
            assets[a] = assets.get(a, 0) + 1
        top = sorted(assets.items(), key=lambda x: x[1], reverse=True)[:5]
        return envelope({"site_id": site_id, "period_months": period_months,
                         "total_workorders": len(wos), "completed": len(completed),
                         "backlog": len(backlog), "overdue": len(overdue),
                         "avg_completion_hrs": avg_hrs, "priority_breakdown": prio,
                         "top_assets_by_wo_count": [{"asset": a, "count": c} for a, c in top]},
                        duration_ms=t_ms(t))


async def get_schedule_calendar(db, site_id: str, date_from: Optional[str] = None,
                                date_to: Optional[str] = None, group_by: str = "date",
                                now: Optional[datetime] = None) -> Dict[str, Any]:
    """Scheduled (non-terminal) work orders in a date window, optionally bucketed by day."""
    with Timer() as t:
        now = now or datetime.now(timezone.utc)
        date_from = date_from or now.strftime("%Y-%m-%d")
        date_to = date_to or (now + timedelta(days=14)).strftime("%Y-%m-%d")
        docs = await db.find({"type": "workorder", "siteid": site_id.upper()}, limit=10000)
        in_win = []
        for w in docs:
            if (w.get("status") or "") in TERMINAL:
                continue
            ss = w.get("schedstart") or w.get("targstartdate")
            if not ss:
                continue
            day = ss[:10]
            if day < date_from or day > date_to:
                continue
            in_win.append(w)
        if group_by == "date":
            buckets: Dict[str, List[Dict]] = {}
            for w in in_win:
                day = (w.get("schedstart") or w.get("targstartdate"))[:10]
                buckets.setdefault(day, []).append(_public(w))
            payload = {"site_id": site_id, "date_from": date_from, "date_to": date_to,
                       "total_scheduled": len(in_win),
                       "by_date": [{"date": d, "count": len(r), "workorders": r}
                                   for d, r in sorted(buckets.items())]}
        else:
            payload = {"site_id": site_id, "date_from": date_from, "date_to": date_to,
                       "total_scheduled": len(in_win), "workorders": [_public(w) for w in in_win]}
        return envelope(payload, duration_ms=t_ms(t), record_count=len(in_win))


async def get_my_assigned_workorders(db, labor_code: str, site_id: Optional[str] = None,
                                     open_only: bool = True) -> Dict[str, Any]:
    """Work orders with a `wplabor` line for the given labor (technician)."""
    with Timer() as t:
        docs = await db.find({"type": "workorder", "wplabor": {"$elemMatch": {"laborcode": labor_code}}},
                             limit=10000)
        out = []
        for w in docs:
            if site_id and (w.get("siteid") or "").upper() != site_id.upper():
                continue
            if open_only and (w.get("status") or "").upper() in TERMINAL:
                continue
            out.append(_public(w))
        return envelope({"labor_code": labor_code, "workorders": out, "totalCount": len(out)},
                        duration_ms=t_ms(t), record_count=len(out))


# --------------------------------------------------------------------------- #
# Write tools
# --------------------------------------------------------------------------- #
async def create_workorder(db, description: str, asset_num: str, site_id: str,
                           priority: int = 3, work_type: str = "CM",
                           reported_by: Optional[str] = None, location: Optional[str] = None,
                           notes: Optional[str] = None, wonum: Optional[str] = None,
                           aob_source: Optional[Dict[str, Any]] = None,
                           now: Optional[datetime] = None) -> Dict[str, Any]:
    """Create a new work order (status WAPPR). Optionally pin `wonum` for reproducible ids
    and attach `aob_source` provenance (the agent/trigger that generated it)."""
    if not description or not asset_num or not site_id:
        return error("description, asset_num, and site_id are required", "VALIDATION_ERROR")
    if not 1 <= priority <= 5:
        return error("priority must be between 1 and 5", "VALIDATION_ERROR")
    if work_type not in WORKTYPES:
        return error(f"work_type must be one of {WORKTYPES}", "VALIDATION_ERROR")

    with Timer() as t:
        now = now or datetime.now(timezone.utc)
        won = wonum or await db.next_wonum(site_id)
        doc: Dict[str, Any] = {
            "_id": _doc_id(site_id, won), "type": "workorder", "schema_version": "1.0.0",
            "wonum": won, "siteid": site_id.upper(), "description": description[:100],
            "assetnum": asset_num, "wopriority": priority, "worktype": work_type,
            "status": "WAPPR", "reportdate": _iso(now),
        }
        if reported_by:
            doc["reportedby"] = reported_by
        if location:
            doc["location"] = location
        if notes:
            doc["description_longdescription"] = notes
        if aob_source:
            doc["aob_source"] = aob_source
        await db.put(doc)
        return envelope(_public(doc), duration_ms=t_ms(t))


# Alias matching the AssetOpsBench WO Agent's tool name.
async def generate_work_order(db, **kwargs) -> Dict[str, Any]:
    """Alias for create_workorder, named to match the AssetOpsBench WO Agent."""
    return await create_workorder(db, **kwargs)


async def update_workorder(db, wonum: str, site_id: str, description: Optional[str] = None,
                           priority: Optional[int] = None, location: Optional[str] = None,
                           asset_num: Optional[str] = None, notes: Optional[str] = None) -> Dict[str, Any]:
    """Update mutable fields on an existing work order."""
    with Timer() as t:
        doc = await db.get(_doc_id(site_id, wonum))
        if not doc:
            return error(f"Work order '{wonum}' not found", "NOT_FOUND")
        if description is not None:
            doc["description"] = description[:100]
        if priority is not None:
            doc["wopriority"] = priority
        if location is not None:
            doc["location"] = location
        if asset_num is not None:
            doc["assetnum"] = asset_num
        if notes is not None:
            doc["description_longdescription"] = notes
        await db.put(doc)
        return envelope(_public(doc), duration_ms=t_ms(t))


async def _change_status(db, wonum: str, site_id: str, new_status: str,
                         extra: Optional[Dict[str, Any]] = None,
                         now: Optional[datetime] = None) -> Dict[str, Any]:
    with Timer() as t:
        doc = await db.get(_doc_id(site_id, wonum))
        if not doc:
            return error(f"Work order '{wonum}' not found", "NOT_FOUND")
        doc["status"] = new_status
        doc["status_date"] = _iso(now or datetime.now(timezone.utc))
        if extra:
            doc.update(extra)
        await db.put(doc)
        return envelope(_public(doc), duration_ms=t_ms(t))


async def approve_workorder(db, wonum: str, site_id: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Approve a work order (status → APPR)."""
    return await _change_status(db, wonum, site_id, "APPR", now=now)


async def cancel_workorder(db, wonum: str, site_id: str, reason: Optional[str] = None,
                           now: Optional[datetime] = None) -> Dict[str, Any]:
    """Cancel a work order (status → CAN)."""
    extra = {"description_longdescription": reason} if reason else None
    return await _change_status(db, wonum, site_id, "CAN", extra=extra, now=now)


async def assign_technician(db, wonum: str, site_id: str, labor_code: str,
                            craft: Optional[str] = None, start_date: Optional[str] = None,
                            hours_planned: float = 8.0, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Append a planned-labor (`wplabor`) line assigning a technician to the work order."""
    if not all([wonum, site_id, labor_code]):
        return error("wonum, site_id, and labor_code are required", "VALIDATION_ERROR")
    with Timer() as t:
        doc = await db.get(_doc_id(site_id, wonum))
        if not doc:
            return error(f"Work order '{wonum}' not found", "NOT_FOUND")
        line: Dict[str, Any] = {"laborcode": labor_code, "laborhrs": hours_planned,
                                "startdate": start_date or _iso(now or datetime.now(timezone.utc))}
        if craft:
            line["craft"] = craft
        doc.setdefault("wplabor", []).append(line)
        await db.put(doc)
        return envelope(_public(doc), duration_ms=t_ms(t))


async def close_workorder(db, wonum: str, site_id: str, actual_hours: float = 0.0,
                          failure_code: Optional[str] = None, resolution_notes: Optional[str] = None,
                          now: Optional[datetime] = None) -> Dict[str, Any]:
    """Close a work order (status → COMP), recording actual hours, failure code, resolution,
    and stamping `actfinish`."""
    now = now or datetime.now(timezone.utc)
    extra: Dict[str, Any] = {"actlabhrs": actual_hours, "actfinish": _iso(now)}
    if failure_code:
        extra["failurecode"] = failure_code
    if resolution_notes:
        extra["description_longdescription"] = resolution_notes
    return await _change_status(db, wonum, site_id, "COMP", extra=extra, now=now)


def t_ms(timer: Timer) -> int:
    # Timer.ms is only set on __exit__; inside the block fall back to 0.
    return getattr(timer, "ms", 0)