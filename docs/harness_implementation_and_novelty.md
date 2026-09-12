# Implementing the harness, and where the novelty is

Companion to `harness_optimization_plan.md`. That document settles the method.
This one settles the code and the contribution.

## Part 1: the implementation

### The hook point already exists

`src/agent/stirrup_agent/workspace_bridge.py` defines

```python
class WorkspaceBridgedMCPToolProvider(MCPToolProvider):
    async def __aenter__(self) -> list[Tool[Any, ToolUseCountMetadata]]:
```

A provider that subclasses `MCPToolProvider` and returns a list of `Tool`
objects from `__aenter__` is exactly the seam an adapter needs. The adapter is
one more subclass in the same position, and it is the third time this session we
have used this pattern: the skill mount wraps the code provider at `__aenter__`,
the preserve wrapper wraps it at `__aexit__`, and now the adapter wraps the MCP
provider at `__aenter__`.

```python
class AdaptedMCPToolProvider(WorkspaceBridgedMCPToolProvider):
    """Client-side adaptation. The servers never change."""

    async def __aenter__(self) -> list[Tool]:
        tools = await super().__aenter__()
        return [adapt(t) for t in tools if not dropped(t)]
```

`adapt` does four things, all of them client-side and all of them on
AutoSaddler's taxonomy:

| Adapter file | Patch subtype | What it changes |
| --- | --- | --- |
| `schemas.py` | Argument Modification | the parameter model the agent sees |
| `descriptions.py` | Tool Description Fix | the docstring the agent reads |
| `preprocess.py` | Argument Modification | arguments, before the call leaves |
| `hooks.py` | PreToolUse Hook | a reminder injected before a named tool |
| `composites.py` | New Tool Addition | a derived tool over existing calls |

`composites.py` is where the highest-accepting patch type lives (83% in
AutoSaddler's Figure 3c). A composite is a new `Tool` whose implementation calls
two or three existing MCP tools and returns a joined result. No server changes,
and it is the single most valuable thing the proposer can write.

### Wiring

One line in `_build_mcp_provider`, gated so the empty adapter is a no-op:

```python
provider_cls = (
    AdaptedMCPToolProvider if self._adapter_dir else WorkspaceBridgedMCPToolProvider
)
```

With `--adapter-dir` absent, behaviour is byte-identical to today. That is the
same guarantee `k0` gives, and it is what makes the baseline arm honest.

### Order of work

1. `AdaptedMCPToolProvider` with an empty adapter, plus a test asserting the
   tool list is unchanged when the adapter is empty
2. one hand-written adapter per subtype, to prove each hook actually reaches the
   agent
3. the journal, policy and validator from the plan
4. the loop

Steps 1 and 2 need no model and no benchmark spend.

## Part 2: the novelty

### The sentence that hands it to you

AutoSaddler, section 3, defining its optimization space:

> we focus on three classes of harness parameters and **do not consider other
> components, such as memory or skill curation, since our setting assumes tasks
> are largely stateless and independent**.

That single sentence excludes, by name, the two things AssetOpsBench now has and
they did not. Their scope exclusion is your contribution.

### Three axes, and the one nobody has moved

| Axis | StarHarness | AutoSaddler | AIDE | AssetOpsBench could |
| --- | --- | --- | --- | --- |
| Model weights | frozen | frozen | frozen | frozen |
| Scaffold code | evolved | evolved | evolved (as the solution) | evolved |
| **Operating knowledge** | listed, never reported | **excluded by name** | n/a | **evolved** |
| **Task statefulness** | stateless | **excluded by name** | stateless | multi-turn dialogs |

### Novelty 1: evolve the knowledge, not the code

Make the artifact under search the **skill library**, not the adapter. The
proposer writes `SKILL.md` files and router entries rather than Python.
Everything else in the plan is unchanged: same mini-batch loop, same paired dev
gate, same guardrails, same entity-value ban.

The claim that makes it a paper rather than a variation:

> **A code patch is bound to one harness. A skill is prose any agent can read.**

StarHarness and AutoSaddler both report cross-*model* transfer. Neither reports
cross-*harness* transfer, and neither can, because their artifact is code written
against one framework's internals. An evolved skill library is a directory of
markdown. Mount it on the Claude runner, the plan-execute runner, the OpenAI
runner, and measure.

AssetOpsBench has seven runners and a skills mount with a K-level control
already built and tested. No other benchmark in this literature is positioned to
run that experiment.

The honest cost: only the Stirrup runner mounts skills today. Cross-harness
transfer needs a mount in at least two more runners. The Claude runner is the
cheap one, since the Claude Agent SDK has native skill support. Budget that as
real work, not a footnote.

The result table nobody else can produce:

| Artifact | Transfers across models | Transfers across harnesses |
| --- | --- | --- |
| Evolved adapter code | yes, both papers show it | no, by construction |
| Evolved skill library | to be measured | **to be measured** |

A negative result is still publishable. If evolved skills do not transfer across
harnesses, that says operating knowledge is more framework-bound than the field
assumes, which is worth knowing.

### Novelty 2: the entity-swap invariance test

All three papers detect overfitting with held-out splits. That is an indirect
instrument: a memorized fact about chillers still helps a held-out chiller task,
so a held-out split under-detects memorization.

AssetOpsBench can do better, because its scenarios are generated from the files
in `shared/`. Re-render the suite with entity names permuted: `Chiller 6`
becomes `Chiller 11`, `MAIN` becomes `WEST`, work-order numbers shift, sensor
names change consistently. Structure identical, every literal different.

- a harness that learned reusable procedure scores the same
- a harness that memorized entities collapses

Report the gap as an **entity-dependence score**. This is a measurement
contribution, it is cheap because the permutation is a script over `shared/`,
and it converts your earlier worry about the harness absorbing your data from a
concern into a number.

It also strengthens Novelty 1: prose skills are more likely to encode entity
literals than code adapters are, so the test is most needed exactly where the
new artifact lives.

### Novelty 3: stateful harness optimization

AutoSaddler scoped out memory because their tasks are "stateless and
independent". Your dialogs are neither. Run the optimization over multi-turn
dialogs and the harness has a new class of thing to learn: cross-turn
conventions, when to reuse an artifact rather than re-retrieve, how to resolve a
referring expression against the turn router.

This composes with the M-levels already built. The natural experiment is a 2x2
of `m0`/`m1` against evolved/baseline harness, asking whether harness
optimization substitutes for mounted history or compounds with it.

I would hold this one back. It is the most novel and the most expensive, and it
depends on the dialog suite growing past one authored scenario.

## Recommendation

Take Novelty 1 as the contribution and Novelty 2 as the instrument that makes it
credible. Together they are one paper: *evolve what the agent knows rather than
how it is wired, and show the result moves between harnesses in a way code
cannot, verified by a test that separates learned procedure from memorized
entities.*

Build the adapter anyway, in Part 1. It is the control arm. Without a code-patch
baseline evolved under the identical protocol, the skill result has nothing to
beat, and "we evolved skills and it helped" is a much weaker claim than "we
evolved skills and code under one protocol, and only the skills moved between
harnesses".

## The question that decides the shape

Does the skill proposer write skills **from scratch**, or does it **edit the
existing `assetopsbench` graph**? From-scratch is a cleaner claim and a much
larger search space. Editing starts from a working library and converges faster,
but every result then carries a hand-written prior that a reviewer will ask
about. My inclination is from-scratch for the headline arm, with the hand-written
library as a second baseline alongside `k0`.
