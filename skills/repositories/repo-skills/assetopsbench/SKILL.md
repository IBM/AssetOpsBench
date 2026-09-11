---
name: assetopsbench
description: "Operates the AssetOpsBench MCP surface: six stdio FastMCP servers holding
  85 tools across asset and sensor discovery, failure-mode reasoning, time-series
  modelling, vibration diagnostics, work-order management and reference catalogs.
  Route here when a request concerns a physical asset at a site, its sensors or
  telemetry, its failure modes, a forecast or anomaly check on its signals, a
  vibration spectrum, or a maintenance work order. Read this before calling any
  tool, so the server that owns the capability is chosen rather than guessed, and
  so the evidence discipline this environment scores on is applied from the first
  call rather than reconstructed afterwards."
disable-model-invocation: true
license: Apache 2.0
metadata:
  disco-role: operating
  capability-family: C1, C2, C12
  asset-class: A0
  leakage-class: ops
  library-version: 0.1.0
---

# AssetOpsBench tool surface

## Purpose

AssetOpsBench exposes an industrial asset operations environment as six stdio
MCP servers holding 85 tools. This graph is the map of that surface: which
server owns which capability, how to reach it, and the evidence discipline that
applies to every answer here.

This is the reference skill graph shipped with the repository. It is complete
and mountable on its own, and it is deliberately small. See `skills/README.md`
for how a larger library is mounted in its place.

## The one thing to get right first

Answers in this environment are judged on the execution record, not only on the
claim. A conclusion that no executed tool call supports is not a weaker answer,
it is an unsupported one, and it scores as one. Two consequences that change
what you do before you have any results:

1. **Retrieve before you assert.** If you cannot name the call that produced a
   number, do not put the number in the answer.
2. **Abstain rather than interpolate.** Reporting that the evidence is
   insufficient is a correct answer when it is true. Filling the gap with a
   plausible value is not a partially correct answer, it is a wrong one that is
   harder to detect.

## Server access

The six servers are launched as stdio subprocesses. The launch contract lives
in `src/mcphub/__init__.py`:

```python
DEFAULT_SERVERS = {n: ["uv", "run", f"{n}-mcp-server"]
                   for n in ["iot", "utilities", "fmsr", "wo", "tsfm", "vibration"]}
```

Before anything else in a fresh environment, prove the surface is reachable and
is the surface this skill documents:

```bash
python scripts/check_servers.py --json
```

It completes the MCP handshake, calls `tools/list`, and asserts that the
documented tool names are the live tool names. A server that fails the handshake
is unavailable, not empty. Do not work around it by guessing values; say the
server is down and stop.

`AOB_READONLY=1` removes the six work-order mutation tools. Check whether it is
set before planning any write, because a plan that ends in a write you cannot
perform is a plan you have to redo.

## Which server owns what

| Server | Tools | Owns |
| --- | ---: | --- |
| `iot` | 12 | Sites, assets, sensors, and telemetry history |
| `fmsr` | 3 | Failure modes and their sensor relationships |
| `tsfm` | 41 | Forecasting, anomaly detection, data quality, recipes and runs |
| `vibration` | 8 | Spectra, envelope analysis, bearing frequencies |
| `wo` | 15 | Work orders: history, distribution, generation and updates |
| `utilities` | 6 | Reference catalogs and lookups |

Full tool-by-tool inventory: `references/server-capability-map.md`.

## Sub-skills

Open one, for the step you are on. Do not read both up front.

| Sub-skill | Open it when |
| --- | --- |
| [`server-routing`](sub-skills/server-routing/SKILL.md) | You know what you need and not which server has it, or a server is refusing, or you are about to write |
| [`evidence-and-abstention`](sub-skills/evidence-and-abstention/SKILL.md) | You are about to state a conclusion, or you suspect the evidence does not reach it |

## Failure modes of this skill

- **It maps the surface, not the domain.** It tells you `vibration` owns
  envelope analysis. It does not tell you whether the sampling rate resolved the
  harmonic you are about to name. That judgement lives in a domain library.
- **The tool counts are pinned to a commit.** If `check_servers.py` reports
  `SKILL_GAP`, the skill is stale and the server is right.

## Stop conditions

Stop and report rather than proceeding if a server fails its handshake, if a
requested asset or sensor does not resolve, or if the only path to an answer is
a value no call returned.
