# harness_opt

An outer loop that searches client-side adapter configurations while the model
weights, the MCP servers and the scenario data stay fixed. Design and
provenance: `docs/harness_optimization_plan.md`.

## What is fixed, and what moves

| | |
| --- | --- |
| model weights | fixed. Nothing here trains anything |
| `src/servers/` | fixed. The servers are the environment |
| `src/couchdb/`, `src/evaluation/` | fixed. Scenarios and scoring are the environment |
| `src/agent/stirrup_agent/adapter/` | **the only surface a patch may change** |
| `src/agent/_prompts.py` | editable |

A candidate is an `adapter.json`. The runner loads it with `--adapter-dir`, and
with no adapter the provider class is returned unchanged, so the baseline arm
runs the code that was already there.

## Modules

| Module | Does |
| --- | --- |
| `splits.py` | group-wise train/dev/reserve by scenario category, pinned to a file |
| `reflect.py` | the paired 2x2 against the parent; the accept rule and the trend |
| `journal.py` | nodes, parents, buggy nodes, the summarized view |
| `policy.py` | which node and which operator, AIDE's policy on a dev metric |
| `validate.py` | scope, spec, no-leak and no-entity-values gates |

## The three decisions worth knowing

**Accept on the paired table, not the mean.** The scorer is deterministic; the
agent is not. `reflect.compare` buckets every dev task against the parent into
fixed / regressed / still-passing / still-failing, and a patch is accepted when
`fixed > regressed`. Comparing means would have the loop accepting noise.

**Measure the noise first.** `reflect.flip_rate` takes several baseline runs and
returns the per-task disagreement between them. Run it before proposing
anything; every accept decision is implicitly compared against that number.

**A failed candidate is a node, not a discard.** Scope violations and malformed
specs are appended with `is_buggy=True`. Dropping them would leave the debug
operator with nothing to select and silently halve the search.

## The entity gate

`validate.check_no_entity_values` separates schema identifiers from data values.
An adapter narrowing a work-order schema has to name the field `siteid`; that is
the interface. Writing `MAIN` into the harness memorizes the corpus. The corpus
is harvested from the *values* columns take under `shared/`, never their names.

## Tests

```bash
uv run pytest harness_opt/tests src/agent/stirrup_agent/adapter/tests -q
```

108 tests, no third-party dependency beyond pytest, so the loop is verifiable
without a model, a container or a live MCP server.
