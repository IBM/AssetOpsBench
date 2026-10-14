# Running the Benchmark with Skills

An agent working this environment brings two things: a model, and whatever it
knows about industrial asset operations. The second is usually implicit, buried
in a system prompt or in whatever the backbone happens to remember about
bearings and chillers. This page makes it explicit, mountable, and reportable as
a level, so a run can say which operating knowledge it had.

The mechanism is a **skill library** copied into the agent's code-execution
workspace, and a **K level** that says whether it was mounted. `K0` is the
unaided baseline and is byte-identical to the behaviour before any of this
existed. `K1` mounts a library. The difference between them, per task, is the
measurement.

Related docs: [running_benchmark.md](running_benchmark.md) for the suite runner
and the leaderboard, [stirrup-agent.md](stirrup-agent.md) for the agent,
[../skills/README.md](../skills/README.md) for the library that ships here and
[../skills/CONTRACT.md](../skills/CONTRACT.md) for writing your own.

---

## Quick start

```bash
# 0. everything from running_benchmark.md first: uv sync, .env, CouchDB, code image

# 1. prove the agent will see the skills, before spending a run
python skills/preflight.py --assetops . --skills skills/repositories

# 2. one scenario, unaided
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k0 \
  --k-level k0 "<scenario prompt>"

# 3. the same scenario, with the library mounted
uv run python -m agent.stirrup_agent.cli --workspace-dir ./ws-k1 \
  --skills-dir skills/repositories --k-level k1 "<scenario prompt>"
```

The preflight is worth the ten seconds. Its check 4 is the one that matters:

```
PASS  4 mount k0         nothing mounted, nothing appended
```

That proves `K0` really is unaided. A contaminated baseline invalidates every
comparison downstream of it, and it fails silently otherwise.

---

## The three K levels

| Level | What happens | Use it for |
| --- | --- | --- |
| `k0` | Mounts nothing, appends nothing to the system prompt | The baseline. This is the default |
| `k1` | Copies the library into the workspace, appends a routing block | The treatment |
| `k1-recovery` | Mounts the library but tells the agent to work unaided first and consult only after a concrete failure | Scoring the library on recovery rather than on substitution |

`--k-level` defaults to `k0`, so nothing changes for anyone who does not pass the
new flags.

`k1-recovery` answers a different question from `k1` and costs another full arm.
Run it only if you intend to report it.

---

## What `--skills-dir` points at

The directory holding **both** `repo-skills/` and `repo-skills-router/`. Not one
or the other: the router is the index into the graphs, and the prompt block names
only the router.

```
skills/repositories/            <- this is the path you pass
  repo-skills/                  the graphs
  repo-skills-router/           the index
```

Two moving parts, both in `src/agent/stirrup_agent/skills_mount.py`:

1. The library is copied into the code-execution workspace, so the agent sees it
   at `/workspace/skills` under the Docker backend and `skills/` locally.
2. A block of about 650 characters is appended to the system prompt, naming the
   router and the routing discipline.

The agent already has a shell, so it reads a `SKILL.md` with `cat` and
progressive disclosure comes free: router, then one graph, then one sub-skill.
Nothing is loaded until it is chosen. **This is why the prompt cost does not grow
with the library**: a one-graph library and a forty-graph library both cost the
same 665 characters up front.

### Using a different library

Change one path. Nothing else, and no code change:

```bash
--skills-dir /path/to/other-library/repositories
```

The library that ships here is a small reference one, complete and mountable,
covering this repository's own tool surface. A larger library, held anywhere,
mounts the same way. Validate any library before you mount it:

```bash
python skills/tools/validate_skills.py --root /path/to/other-library/repositories
```

---

## A suite run, one arm at a time

`--skills-dir` and `--k-level` are threaded through
`benchmark.scenario_suite_runner`, so a suite run takes them directly:

```bash
MODEL="litellm_proxy/aws/claude-opus-5"
SKILLS=skills/repositories

# K0
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids lite --scenario-root benchmarks/scenario_suite \
  --agent_name stirrup_agent --model-id "$MODEL" --reasoning-effort high \
  --k-level k0 \
  --trajectory-root runs/k0/assetopsbench-trajectories \
  --reports-root    runs/k0/assetopsbench-reports \
  --stirrup-workspace-root runs/k0/ws --preserve-workspaces

# K1, identical except for the two skill flags and the output roots
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids lite --scenario-root benchmarks/scenario_suite \
  --agent_name stirrup_agent --model-id "$MODEL" --reasoning-effort high \
  --skills-dir "$SKILLS" --k-level k1 \
  --trajectory-root runs/k1/assetopsbench-trajectories \
  --reports-root    runs/k1/assetopsbench-reports \
  --stirrup-workspace-root runs/k1/ws --preserve-workspaces
```

> **Give each arm its own output roots.** The suite runner names trajectory files
> by scenario id alone, so two arms sharing a root means the second silently
> overwrites the first, and the pairing below then compares an arm against
> itself. This is the single easiest way to waste a suite of runs.

Keep everything else identical between arms: model, reasoning effort,
temperature, scenario selector, and **the commit**. A model or a code change
between arms is a confound the analysis cannot detect, because both arms still
look structurally fine.

### Check the mount reached the agent, once

`--preserve-workspaces` exists for this. Three commands, one time, and you never
again wonder:

```bash
ls runs/k0/ws/stirrup_agent/*/*/skills 2>/dev/null    # must be empty
ls runs/k1/ws/stirrup_agent/*/*/skills                # repo-skills, repo-skills-router
grep -l "repo-skills" runs/k1/assetopsbench-trajectories/*/*/*.json
```

---

## Measuring the difference

Per task, `s(t) = score_K1(t) - score_K0(t)`. Two tools do the work, and neither
needs anything instrumented: the benchmark already writes the score and the
operational metrics to `_aggregate.json`, and the trajectory already records
which `SKILL.md` files the agent opened.

### 1. Build a run manifest

```bash
python skills/tools/build_run_manifest.py --k-level k0 \
  --reports-root    runs/k0/assetopsbench-reports \
  --trajectory-root runs/k0/assetopsbench-trajectories \
  --out runs/manifest.jsonl --expect-no-skills

python skills/tools/build_run_manifest.py --k-level k1 \
  --reports-root    runs/k1/assetopsbench-reports \
  --trajectory-root runs/k1/assetopsbench-trajectories \
  --out runs/manifest.jsonl --append --expect-skills
```

`--expect-no-skills` and `--expect-skills` check the arm label against what the
trajectories actually show. A mislabelled arm produces a clean-looking manifest
and a meaningless result, which is the kind of mistake you find months later.

Add `--asset-class-map map.json`, a small JSON file mapping scenario id to asset
class, to enable the per-class breakdown.

### 2. Run the analysis

```bash
python skills/tools/gate5_counterfactual.py --runs runs/manifest.jsonl \
  --per-graph --emit gate5-admission.json
```

What it reports, and why each part is there:

| Output | Why |
| --- | --- |
| Mean `s`, paired bootstrap interval, sign test | The resampling unit is the task, not the run, so repetitions of one scenario do not masquerade as independent observations |
| Regression count and rate, with a budget | Never netted into the mean. A library that helps on average while poisoning one asset class is worse than no library |
| Per asset class | Where that poisoning becomes visible behind a positive mean |
| Per graph | Restricted to the tasks where the agent actually opened that graph, corrected across graphs by Benjamini-Hochberg |
| Minimum detectable effect | So a null reads as "no effect" or "not enough runs", which are different findings |
| Spearman of `s` against extra tokens, steps, tool calls | Pre-empts "you just spent more compute" |
| Contamination check on the recorded `k0` runs | The preflight proves the k0 *code path* mounts nothing; this proves the k0 *runs that were scored* consulted nothing |

The verdict is one of `ADMITTED`, `NOT_SHOWN_TO_HELP`, `UNDERPOWERED`,
`HARMFUL` or `VOID`. Exit code is 0 only when the effect is admitted and nothing
else failed, so it can sit in CI as a release gate once a baseline exists.

`python skills/tools/gate5_counterfactual.py --self-test` plants known effects
and checks they are recovered, that a null library is refused, and that a
contaminated baseline voids the run. Run it before spending a suite on it.

### How many tasks you need

Minimum detectable mean `s` at 80 percent power, two-sided alpha 0.05:

| Paired tasks | sd 0.15 | sd 0.25 | sd 0.35 |
| --- | ---: | ---: | ---: |
| 3 (`open`) | 0.243 | 0.404 | 0.566 |
| 50 (`lite`) | 0.059 | 0.099 | 0.139 |
| 215 (`all`) | 0.029 | 0.048 | 0.067 |

The per-graph table is the harder constraint: a graph the agent opens on 20 of
215 tasks needs roughly a 0.16 effect to clear its own interval, so expect
`INSUFFICIENT_POWER` on many graphs even at full scale. That is reported rather
than hidden, and how often each graph was opened is itself a finding about the
suite's coverage.

Start with `open` (3 scenarios) to shake out the plumbing. It cannot answer the
research question and is not meant to; a verdict of `UNDERPOWERED` there is the
correct result.

---

## Before every evaluation

```bash
# frontmatter, licence, self-containment, routing metadata, industrial axes
python skills/tools/validate_skills.py --root skills/repositories

# the leakage audit, which needs your answer set
python skills/tools/validate_skills.py --root skills/repositories \
  --answers-hf ibm-research/AssetOpsBench
```

> **Treat a leakage hit as blocking.** A skill library sits closer to the answers
> than anything else an agent reads. The audit fails any eight-word sequence
> shared between a skill and the answer set, and names the scenario each hit came
> from so it can be triaged rather than argued with. Pointing it at the checkout
> proves nothing: `benchmarks/scenario_suite/*.yaml` holds scenario **ids** only.
> Use `--answers-hf` for the published dataset, or `--answers-dir` for whatever
> your harness exports.

A `leakage-class: solution` skill fails outright rather than warning.

---

## Recording a run

Put the **K level** and the **library version** in every results row, beside the
model id. A library bump moves the leaderboard exactly as a model change does,
and a run that does not record which library it read cannot be compared with one
that read another. The library version is `metadata.library-version` in the
frontmatter, and the taxonomy it was routed against is `taxonomy_sha256` in each
graph's `repo-routing-metadata.json`.

Keep the transcript. Per-graph attribution reads it for the skill paths the agent
opened, so without it every other part of the analysis still works and the
per-graph table is empty.

---

## What this does not tell you

Mounting a library and measuring a delta says whether *that* library helped *this*
suite at *this* power. It does not say operating knowledge helps in general, and
it says nothing at all about a graph no run ever opened.

Until a paired suite has actually run, the honest description of any library is
**constructed and gated**, not shown to help. The gate exists so that claim can
become a measured one; running it is what changes the wording.
