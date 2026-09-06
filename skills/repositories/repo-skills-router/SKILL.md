---
name: repo-skills-router
description: "Routes a request to the skill graph that owns the capability, narrowing area, then family, then graph, then sub-skill. Read this first, before opening any graph, so that only the relevant branch is loaded."
license: Apache 2.0
metadata:
  disco-role: operating
---

# Skill router

## Purpose

Narrow before you load. Open an area page, then the graph it names, then that
graph's sub-skill for the step you are on. Read one sub-skill at a time, and
open a reference file only when a sub-skill points at it.

This discipline matters more as the library grows. The routing cost is one page
whether the library holds one graph or forty, which is the whole reason the
prompt names this file and nothing else.

## Areas

| Area | Graphs |
| --- | ---: |
| [Industrial asset operations](references/areas/industrial-asset-operations.md) | 1 |

## Start here

Read [`references/entry.md`](references/entry.md). It is one page: what this
library covers, the route from what a request asks for to the graph that answers
it, and the rule for when to call an MCP tool versus when to run code in the
workspace.

## What this library is

This is the reference library shipped in the AssetOpsBench repository. It holds
one graph, `assetopsbench`, covering the benchmark's own tool surface and the
evidence discipline it scores on. It is complete and mountable as it stands.

It is also the worked example for the skill contract. A larger library, public
or private, mounts in exactly the same way and replaces this one: see
`skills/README.md` and `skills/CONTRACT.md`.
