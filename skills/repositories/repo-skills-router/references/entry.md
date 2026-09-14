# Entry page

One page. Read this before opening a graph.

## What this library covers

One graph, `assetopsbench`: which of the six MCP servers owns which capability,
and the evidence discipline this environment scores on. It does not cover
domain judgement, which is what a larger library adds.

## Route

| The request is about | Open |
| --- | --- |
| Which server or tool to use, or a server refusing | `assetopsbench` then `server-routing` |
| Whether an answer is supportable, or an underspecified request | `assetopsbench` then `evidence-and-abstention` |
| Anything else | Nothing here covers it. Say so rather than stretching a skill to fit |

That last row is not filler. A library that always has an answer is a library
that is guessing, and routing to a graph that does not cover the step is worse
than routing to nothing, because it lends unearned confidence.

## MCP tool or code workspace

Call an MCP tool when the environment holds the thing: assets, sensors,
telemetry, failure modes, spectra, work orders, runs.

Use the code workspace when the step is computation over things you already
retrieved: arithmetic across two results, a unit conversion, a statistic no
server exposes, a plot. Doing it in code is correct and it is recorded.

Do neither, and say so, when the step needs a value that no call returned and no
computation can produce. That is an abstention, and it is a scored outcome here
rather than a failure to answer.
