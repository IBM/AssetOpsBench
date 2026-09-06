# Skills

Operating knowledge for agents working this benchmark, mounted into the agent's
workspace and reported as a controlled variable rather than baked into a prompt.

This directory holds two separate things, and the distinction is the point:

1. **The interface.** The mount, the K-level control, the skill contract, the
   validator and the router. All public, all in this repository.
2. **A library.** One reference graph, `assetopsbench`, covering this
   repository's own tool surface. Complete and mountable, deliberately small.

A larger library, held anywhere, mounts through the same interface with no code
change. That is what makes the interface worth publishing on its own.

## The split, and why it mirrors the scenario split

The scenario suite in this repository is public; a held-out suite is not. Skills
work the same way, for the same reason.

| | Public, here | Held out |
| --- | --- | --- |
| Scenarios | The released suite | The evaluation suite |
| Skills | The interface, and the `assetopsbench` graph | A larger domain library |

A skill library is an experimental condition. If the library an agent reads is
published alongside the tasks it is scored on, then the library can be tuned to
the tasks, and a result no longer measures whether operating knowledge helps. It
measures whether that knowledge was fitted to that suite. Holding a library out
is the same control as holding scenarios out, and it is why `--k-level` reports
which condition a run used rather than leaving it implicit.

Nothing about the mechanism is secret. Anyone can build a library against the
contract below and run the same three arms.

## Running it

```bash
# K0: unaided baseline. Mounts nothing, appends nothing to the prompt.
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k0 \
  --k-level k0 "<scenario prompt>"

# K1: this repository's reference library
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k1 \
  --skills-dir skills/repositories --k-level k1 "<scenario prompt>"

# K1 with a different library: change one path, nothing else
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k1 \
  --skills-dir /path/to/other-library/repositories --k-level k1 "<scenario prompt>"

# K1-recovery: unaided first, skills consulted only after a concrete failure
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k1r \
  --skills-dir skills/repositories --k-level k1-recovery "<scenario prompt>"
```

`--skills-dir` points at the directory holding **both** `repo-skills/` and
`repo-skills-router/`. It is not one or the other: the router is the index into
the graphs, and the prompt block names only the router.

**K0 is byte-identical to the behaviour before any of this existed.** It mounts
nothing and appends nothing, which is what makes the comparison honest. Record
the K level and the library version in every results row; a library change moves
the leaderboard exactly as a model change does.

## How a skill reaches the agent

Two moving parts, both in `src/agent/stirrup_agent/skills_mount.py`:

1. The library is copied into the code-execution workspace, so the agent sees it
   at `/workspace/skills` under the Docker backend and `skills/` locally.
2. A short block is appended to the system prompt naming the router and the
   routing discipline. It is about 650 characters and it names one file, not the
   library, because the collection is routed rather than enumerated.

The agent already has a shell, so it reads a `SKILL.md` with `cat` and
progressive disclosure comes free: router, then one graph, then one sub-skill.
Nothing is loaded until it is chosen. This is why a library of forty graphs
costs the same prompt budget as a library of one.

## Checking a library before you spend a run

```bash
python skills/tools/validate_skills.py --root skills/repositories
```

Frontmatter contract, per-tree licence consistency, self-containment, and a
leakage audit. Add `--answers` to point the leakage half at your answer set;
without it that half does not run and says so.

The leakage check exists because a skill library sits closer to the answers than
anything else an agent reads. `leakage-class: solution` fails outright, and any
eight-word sequence shared between a skill and the answer set is a failure that
names the scenario it came from.

## Contributing a graph

`CONTRACT.md` has the layout, the frontmatter, and the rules. The short version:
install or clone what you are describing and read it, rather than writing from
memory; make every script a gate that refuses something, with a `--self-test`
that proves it refuses; and lead each sub-skill with the mistake it prevents,
because that is the part a capable model does not already know.
