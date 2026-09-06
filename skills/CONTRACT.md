# Skill contract

What a skill graph must look like to mount and validate here. The graph in
`repositories/repo-skills/assetopsbench/` is a worked example of every rule
below; read it alongside this file.

The format follows the AREX repository-skill contract, with three additions the
physical-asset setting forces: an asset-class axis alongside the capability
axis, a leakage class, and a rule that a script must refuse something.

## Layout

```
repositories/
  repo-skills-router/
    SKILL.md                          the index; the prompt names this file
    references/entry.md               one page: what the library covers, and the route
    references/areas/<area>.md        one page per area
  repo-skills/
    <graph-id>/
      SKILL.md                        root router, 80 to 150 lines with frontmatter
      references/
        repo-provenance.md            required, schema below
        repo-routing-metadata.json    required, schema below
        <evidence>.md                 whatever backs the numbers in the skill
      scripts/<gate>.py               0 to 2 graph-level gates
      sub-skills/<sub-id>/
        SKILL.md                      80 to 250 lines with frontmatter
        scripts/<gate>.py             usually one
```

Every relative link must resolve inside the library. Sub-skill ids are unique
across the whole library, not just within a graph.

## Frontmatter

Required on every `SKILL.md`:

```yaml
---
name: <exactly the directory name, [a-z0-9][a-z0-9-]{0,63}>
description: "<double-quoted; may wrap onto continuation lines indented two
  spaces. Say when to route here, in operator language, naming the concrete
  triggers. 60 to 140 words.>"
disable-model-invocation: true
license: <one licence, identical across every SKILL.md in this graph>
metadata:
  disco-role: operating
  capability-family: <comma-separated, from C1..C12 below>
  asset-class: <comma-separated, from the list below>
  leakage-class: ops
  library-version: 0.1.0
---
```

The router's own `SKILL.md` is the exception: it must not set
`disable-model-invocation`, because it is the file the agent is told to open.

**Capability families.** C1 asset and sensor discovery; C2 time-series retrieval
and conditioning; C3 data-quality triage and instrument faults; C4 signal
processing and vibration; C5 anomaly and change point; C6 forecasting; C7
failure-mode reasoning and sensor mapping; C8 health, degradation and RUL; C9
root-cause isolation and diagnostic chaining; C10 maintenance planning and work
orders; C11 control, setpoint and energy efficiency; C12 evidence assembly and
reporting.

**Asset classes.** `A0` (asset-agnostic), `chiller-hvac`, `ahu`, `pumps`,
`motors-drives`, `fans-blowers`, `compressors`, `bearings-gearboxes`,
`wind-turbine`, `transformers-electrical`.

Two axes rather than one, because a capability and the machine it is applied to
come apart: envelope analysis is a capability, a gearbox is an asset, and the
useful skill lives at the intersection. A single axis forces either twelve
bloated skills or a hundred duplicated ones.

**`leakage-class`** is `ops` for anything that ships. `solution` marks a skill
derived from answers, and the validator fails it outright rather than warning.

## `references/repo-routing-metadata.json`

```json
{
  "schema_version": "2.0",
  "repo_id": "<org/repo, or practice/<graph-id> for a graph with no upstream>",
  "skill_id": "<graph-id>",
  "taxonomy_sha256": "sha256:<64 lowercase hex characters>",
  "routing_status": "classified",
  "assignments": [{ "area": "<area>", "family": "<family>" }]
}
```

The router's area pages are generated from these files, so a graph cannot be
routable and undeclared, or declared and unroutable. `taxonomy_sha256` pins
which taxonomy version the assignment was made against; without it a library can
be re-routed silently and two runs that read "the same" library stop being
comparable.

**Write the digest in URI form**, `sha256:` followed by the hex, the same shape
OCI image references and Subresource Integrity use. This is not decoration. A
bare 64-character hex string is indistinguishable from a credential to an
entropy scanner: this repository runs `detect-secrets`, whose
`HexHighEntropyString` plugin flags a bare digest at 64, 32 and even 16
characters and blocks the commit. The prefix clears every scanner tested and
names the algorithm at the point of use, so it is the better representation
regardless of the scanner. The validator enforces the form and says so by name
if it finds a bare digest.

## `references/repo-provenance.md`

Opens with `    schema: disco.repo-provenance.v1`, then the fields shown in the
example graph: `graph_kind`, the sources you actually read, `inspection_method`,
`license`. Then two sections that carry the weight:

- **Evidence.** Every API you cite, with its real signature, read from the
  installed distribution or cloned source during construction. Then how any
  measured result was produced: the generator, the sample counts, where the
  numbers live.
- **Excluded.** What you did not consult and why. Benchmark payloads. Packages
  that failed to install and what you used instead. Standards text.

## Rules the validator enforces

1. **No absolute paths.** No `/home/...`, no `/Users/...`, no `site-packages`,
   no environment activation. A library is copied into a fresh workspace on
   every run and must work there.
2. **No benchmark leakage.** Nothing derived from scenario payloads, scorer
   logic, ground truth or expected outputs. Exclude at gather time; auditing
   leakage out of a finished skill is strictly worse than never letting it in.
3. **Line limits.** Root 80 to 150, sub-skill 80 to 250. Detail goes to
   `references/`, which is loaded only when a sub-skill points at it.
4. **One licence per graph.** A library may span licences; a single graph may
   not.

## Rules the validator cannot enforce, and which matter more

**Never write an API you have not verified.** Install the package in a throwaway
virtualenv and introspect it, or clone and read the source. Record the exact
version. This single rule is where most of a library's value comes from, and
skipping it produces something that reads like a README and is wrong in the
specifics.

**Never write a number you did not measure.** If a skill says a method loses
three percent, someone computed three percent in the session that wrote it. An
unmeasured quantitative claim is the worst thing a library can ship, because a
reader will check it and everything else becomes suspect at once.

**A script is a gate, not a demo.** It takes a claim or a computed result and
returns a pass or a named rejection. Ship it with a `--self-test` that builds
both a passing and a failing case, and make the failing case the input that
would otherwise have slipped through. A gate that passes everything is not a
gate, and a self-test written to confirm the gate's intent rather than probe its
boundary will not tell you which one you have.

**Every script must respond to `--help` on a bare interpreter.** Import optional
dependencies inside the function that needs them and exit with a one-line
`install X` message, never a traceback.

**Lead with the mistake.** Each sub-skill opens with the error it prevents, then
the procedure, then the gate. State the precondition under which the method
stops working. That precondition is the part a capable model does not already
know, and it is the reason the skill exists.
