---
name: server-routing
description: "Chooses the AssetOpsBench MCP server that owns a capability before any
  tool is called, and handles the three situations where a naive choice goes wrong: a
  capability that looks like it belongs to one server and is served by another, a
  server that fails its handshake, and a step that mutates state when the environment
  is read-only. Open this when you know what you need but not where it lives, when a
  call returns an ErrorResult you did not expect, when you are about to write a work
  order or a failure mode, or when a step has no tool at all and belongs in the code
  workspace instead."
disable-model-invocation: true
license: Apache 2.0
metadata:
  disco-role: operating
  capability-family: C1, C2
  asset-class: A0
  leakage-class: ops
  library-version: 0.1.0
---

# Routing to the server that owns the capability

## The mistake this prevents

Guessing the server from the word in the request. "Show me the vibration on
P-101" contains the word vibration, and the first call is almost always an `iot`
call, because `vibration` operates on a signal you have not retrieved yet. The
servers are split by **what they hold**, not by what the question is about, and
those come apart constantly.

## Preconditions

- [ ] `python scripts/check_servers.py --json` passed, or you know which servers
      are down.
- [ ] You know whether `AOB_READONLY=1` is set.
- [ ] You have a site and an asset identifier, or your first call is the one that
      resolves them.

## Procedure

1. **Resolve identity before capability.** Nothing downstream works on an asset
   you have not resolved. `iot.sites()`, then `iot.asset_ids(site_name)`, then
   `iot.asset_detail(site_name, asset_id)`. A request naming an asset in prose is
   not a resolved identifier; assets have registry ids and prose names are not
   guaranteed to match them.

2. **Route by what is held, using this table.**

   | You need | Server | Note |
   | --- | --- | --- |
   | What assets and sensors exist | `iot` | `installed_sensors` is the registry, `measured_sensors` is the stream. They disagree more often than you expect, and the disagreement is itself a finding |
   | Raw telemetry over a window | `iot` | `history` is paged; `stream_extent` first, so you know what you are asking for |
   | Failure modes for an asset class | `fmsr` | `get_failure_modes` reads, `generate_failure_modes` invents. Do not confuse them in an answer |
   | Forecast, anomaly, data quality, a run | `tsfm` | The largest server, 41 tools. It owns the analysis lifecycle, not just models |
   | A spectrum or envelope | `vibration` | Operates on a signal you supply, so an `iot` retrieval comes first |
   | Work-order history or a new order | `wo` | Six of its fifteen tools mutate |
   | A catalog or lookup | `utilities` | Reference data, not asset data |

3. **Check the read half before the write half.** Every write in this
   environment has a read that should precede it. Generating a work order
   without having read the asset's work-order history produces an order that
   duplicates one already open, and nothing in the tool surface will stop you.

4. **Handle a refusing server as a stop, not a detour.** Every tool returns
   `Union[Result, ErrorResult]`. An `ErrorResult` is information: it usually
   means the identifier did not resolve. A failed handshake is different and
   means the server is not running. Neither is a licence to supply the value
   yourself.

5. **Recognise the steps that have no tool.** Some work has no MCP tool and
   belongs in the code workspace: arithmetic across two retrievals, a unit
   conversion, a plot, a statistic the server does not compute. Doing it in code
   is correct. Asserting it without doing it anywhere is not.

## Interpretation

| Situation | What it means | Do this |
| --- | --- | --- |
| `installed_sensors` lists a tag `measured_sensors` does not | The registry claims a sensor the stream never reports | Report the gap. It often explains why a mode is undiagnosable |
| `stream_extent` returns a span shorter than the window asked for | The data does not cover the question | Narrow the claim to the covered span, or say so |
| A tool returns `ErrorResult` on a name from the request | The prose name is not the registry id | Resolve through `iot`, do not retry with variants |
| `AOB_READONLY=1` and the task needs a write | The environment cannot complete the task | Produce the plan and say the write was not performed |

## Failure modes of this skill

- **It routes, it does not sequence.** Knowing that `tsfm` owns anomaly
  detection does not tell you that a data-quality pass belongs before it. Order
  of operations is a workflow concern.
- **Tool counts are pinned to a commit.** Trust `check_servers.py` over this
  file when they disagree.

## Stop conditions

Stop and report if a server fails its handshake, if an identifier will not
resolve after being looked up through `iot`, or if the remaining path to the
answer requires a value that no call returned and no code step can compute.
