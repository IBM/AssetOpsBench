# Harness optimization on AssetOpsBench: final plan

Three papers, one method. This is the decision-complete version, revised after a
review pass that found five things wrong with the previous draft. Those are
marked **[fixed]**.

- **AIDE** (Weco AI, arXiv 2502.13138). Tree search over solution code.
  [Code is public and MIT licensed.](https://github.com/WecoAI/aideml) On
  MLE-Bench it reaches a 16.9% any-medal rate with o1-preview against
  OpenHands' 4.4%, and on RE-Bench it beat human experts inside six-hour
  windows. Reported cost is $0.50 to $2.50 per task.
- **StarHarness** (ServiceNow, arXiv 2608.24804). Harness evolution with
  stratified search/selection/holdout. Code "coming soon".
- **AutoSaddler** (Microsoft, arXiv 2608.23041). Harness optimization as
  mini-batch offline learning. Code "will be available".

Locked decisions: **client-side adapter only** (`src/servers/` frozen), and
**evolve on open scenarios** accepting that they differ from the held-out suite.

## The commonality

All three are the same algorithm. An outer loop searches a space of artifacts
where each evaluation is expensive and stochastic, guided by an LLM that reads
failures.

| | AIDE | StarHarness | AutoSaddler |
| --- | --- | --- | --- |
| Artifact searched | a solution script | a harness patch chain | a harness |
| Memory | Journal (tree) | ledger, frontier or tree | EvoDAG |
| Operators | draft, debug, improve | explore, draft, debug, merge, improve | 9 typed patch subtypes |
| Selection | greedy best, 50% debug | hill climb, or best surviving node | recombine any subset via EvoDAG |
| Cheap gate | does the code run | scope, imports, smoke, single-task test flip | re-run on the same mini-batch |
| Expensive gate | validation metric | hidden selection set | dev set |
| Sealed set | Kaggle test | holdout, opened once | test split, opened once |
| Memory exposure | summarization operator | ledger, read on demand | `evo-dag` query CLI |

StarHarness's tree operators are AIDE's vocabulary with one addition: "explore a
failure pattern, *draft* a patch, *debug* a failed candidate, *merge* two
compatible nodes, or *improve* an existing node." The lineage is explicit.

Four agreements worth taking as settled:

**Two gates, never one.** Every system spends most of its budget rejecting
candidates cheaply. Nobody runs the full evaluation on an unvetted candidate.

**Repair is half the work.** AIDE spends 50% of steps debugging by default.
StarHarness has a debug operator. AutoSaddler tracks still-failing separately
from regressed. Budget for fixing candidates that crash, not only for proposing
new ones.

**The memory is the optimizer.** All three keep scored history and select from
it rather than walking forward from the current best. AutoSaddler states the
analogy directly: EvoDAG is momentum.

**Never serialize the memory into the prompt.** AIDE has a summarization
operator that extracts performance metrics, hyperparameter settings and
debugging hints instead of appending full history, explicitly to prevent prompt
saturation. AutoSaddler refuses to serialize the DAG and exposes the `evo-dag`
CLI instead. StarHarness has the proposer read the ledger rather than receive
it. This is the routing discipline your skills mount already enforces, arrived
at independently by three groups. Build the proposer against a query interface
from the start.

## The one place they disagree, and it decides your design

**AIDE is allowed to overfit. The harness papers are not.**

AIDE formalizes its objective as `s* = argmax_s h(s)` over the space of Python
scripts, where `h` is a stateless objective such as validation accuracy. Its
artifact *is* the solution to one task, so fitting that task is the goal.

Note precisely how AIDE handles the holdout: it keeps an internal train/val
split for scoring nodes, and the external test split is applied *by the
evaluators, outside the search loop*. The search itself has no generalization
gate. A harness must work on tasks it never saw, so that same greedy policy is
precisely the failure mode, and the gate has to move inside the loop.

AutoSaddler measured the cost. Removing generalization-aware selection drops
GAIA2 from 62.0 to 50.6, the largest of their three ablations by a factor of
two. And the reason is not worse patches: both settings reach *similar fix
rates*. The difference is regressions, trending at -0.24 pp per iteration with
the dev gate and +0.16 pp without it.

## Noise, and the accept rule [fixed]

The previous draft said "accept if the dev score improves", and justified it by
calling the evaluation reproducible because `static_json` is deterministic. That
conflated two things. The **scorer** is deterministic. The **agent** is not: the
same harness on the same task can pass one run and fail the next. There is a
`--temperature` flag and no seed anywhere in the runner. Comparing two dev means
without knowing the run-to-run spread would have had the loop accepting noise.

Three corrections.

**Measure the noise before anything else.** Run the baseline three times over
the dev categories and record per-task pass/fail each time. The quantity that
matters is the per-task flip rate between identical runs. Everything downstream
is calibrated against it, and if that rate is high the approach needs larger
sets before it needs a cleverer proposer.

**Gate on the paired table, not the mean.** `ScorerResult.passed` is per task, so
every dev evaluation yields a 2x2 against the parent harness:

|  | patch passes | patch fails |
| --- | --- | --- |
| **parent passes** | still-passing | **regressed** |
| **parent fails** | **fixed** | still-failing |

That is AutoSaddler's reflection bucketing and McNemar's table at once. Accept
when `fixed > regressed`. A paired comparison cancels per-task difficulty, which
a difference of means does not, and it costs nothing extra because the buckets
were already in the plan.

**Do not demand per-patch significance; you cannot afford it.** On a 32-task dev
set the exact binomial says a patch needs about 6 fixes and 0 regressions, or 9
and 1, to clear p<0.05. Almost no real patch clears that bar, so a significance
gate would reject nearly everything. The honest protocol is a weak per-patch
gate (`fixed > regressed`) plus a claim of significance on the **cumulative
trend** across iterations, which is exactly what AutoSaddler's Figure 4 reports
and not what their per-patch rule does.

Two consequences. Enlarge dev: `fmsr` + `fcc` gives 59 tasks instead of 32, and
dev runs on only a minority of iterations so it is affordable. And run the
search at a low fixed temperature to suppress variance, then measure the final
harness at production temperature and report both.

## Splits [fixed]

The previous draft picked train and dev by task count. That is the wrong
criterion. A category the baseline already passes has no headroom to evolve
against, and a category with one failure mode teaches one lesson.

**Choose by baseline headroom and failure-mode diversity, after the Phase 0
baseline run.** The split below is a starting point, not a decision:

| Split | Categories | Tasks |
| --- | --- | --- |
| train | `wosr`, `car` | 116 |
| dev | `fmsr`, `fcc` | 59 |
| reserve | `tsfm`, `health` | 40 |
| test | held-out suite | opened once |

Swap categories between train and dev if the baseline shows the headroom sits
elsewhere. Keep the reserve for a second rotation, which turns one protocol into
several measurements.

## What to take from each

| From | Take | Why |
| --- | --- | --- |
| AIDE | Journal/Node structure, the three operators, the search policy, the tested defaults | The only public implementation, MIT licensed, and the simplest |
| AutoSaddler | mini-batch loop, patch taxonomy, phased schedule, reflection buckets | Cheapest per iteration, and the only published patch action space |
| StarHarness | frozen environment, forbidden-change classes, group-wise holdout opened once | The only paper that takes leakage seriously |

## Architecture

```
harness_opt/
  journal.py     Node + Journal, lifted from AIDE, metric -> dev paired table
  operators.py   draft, debug, improve; patch typed per AutoSaddler Table 1
  policy.py      AIDE search_policy, greedy replaced by best-on-dev
  validate.py    scope, imports, smoke, no-leak, no-entity-values
  splits.py      group-wise train/dev/test by scenario category
  loop.py        the iteration
  reflect.py     the 2x2 and the trend
```

The editable surface, and the only thing a patch may touch:

```
src/agent/stirrup_agent/adapter/
  schemas.py        narrowed and renamed arguments
  preprocess.py     strip nulls, coerce types, fill defaults
  descriptions.py   tool descriptions the agent sees
  composites.py     derived tools over existing MCP calls
  hooks.py          pre-tool-call reminders
src/agent/_prompts.py
```

Scope enforcement is one line: the diff touches only those paths.

**h0 is an empty, pass-through adapter.** That matters for the same reason `k0`
did: with the adapter empty, behaviour is byte-identical to today, so the
baseline arm is the thing that was already there.

**Nodes store adapter snapshots, not diff chains** [fixed]. The adapter is five
small files. Copying the whole directory per node costs nothing and removes an
entire class of failure: patches that no longer apply once the frontier moves,
three-way merge conflicts, ordering bugs. Compute diffs for display and for the
proposer's context, but let the node own a complete directory. A merge operator,
if you ever add one, becomes a file-level merge rather than a patch rebase.

## The loop

```
for iteration n until budget exhausted:
    node   = policy.select(journal)
    patch  = operators.apply(node, traces)     # draft | debug | improve, typed
    if not validate(patch):
        journal.append(Node(patch, is_buggy=True, analysis=reason))   # [fixed]
        continue
    batch  = sample(D_train, B)
    if not batch_improves(patch, node, batch): record and continue
    table  = paired_dev(patch, node, D_dev)    # fixed / regressed / ...
    journal.append(Node(patch, metric=table, is_buggy=False))
    reflect(table)
```

**Validator failures must become nodes** [fixed]. The previous draft rejected
them outright, which quietly broke the debug operator: with no buggy nodes in
the journal, `debug_prob=0.5` selects nothing and the loop degenerates to
draft-and-improve. In AIDE a buggy node is the *input* to debugging, not a
discard. Record scope violations, import errors and smoke-test crashes as nodes
with `is_buggy=True` and the reason in `analysis`.

**"Draft" needs a definition here** [fixed]. In AIDE it means writing a solution
from scratch, which is meaningless for a harness whose baseline already works.
Read it as: propose an independent patch against **h0**, not against the current
best. `num_drafts=5` then means five independent first-generation attempts
before the policy starts improving any of them, preserving AIDE's intent of
seeding diverse starting points.

## Starting parameters

| Parameter | Value | Source |
| --- | --- | --- |
| `num_drafts` | 5 | AIDE config |
| `debug_prob` | 0.5 | AIDE config |
| `max_debug_depth` | 3 | AIDE config |
| mini-batch `B` | 8 | estimate; AutoSaddler does not report it |
| capability-phase epochs `E` | 1 | AutoSaddler, "works well" |
| search temperature | low, fixed | to suppress the variance measured in Phase 0 |

Phase transition at `k = E * ceil(|D_train| / B)`: capability patches before it,
steering patches after. Removing only that schedule cost AutoSaddler 5.9 points,
more than removing deep diagnosis.

## Cost, honestly [fixed]

The previous draft implied a four-fold saving over StarHarness. Recomputed at
B=8, dev=32, 20 iterations, counting both the parent's and the patch's batch
runs:

| Batch-gate pass rate | No cache | With per-task cache |
| --- | --- | --- |
| 20% | 448 runs | 288 runs |
| 30% | 512 runs | 352 runs |
| 50% | 640 runs | 480 runs |

Against roughly 1,080 for a StarHarness-style 54-task selection set. The real
saving is two- to three-fold, not four, and the cache buys the last third.
**Cache per-task results keyed by (adapter snapshot hash, task id)**; most
patches change behaviour on a handful of tasks, so the parent's batch score is
usually already known. Widening dev to 59 raises the dev column proportionally,
which is the price of the noise fix and worth paying.

## Guardrails

StarHarness's four forbidden classes, plus the fifth your data needs:

1. branching on task IDs or hard-coded answers
2. verifier or assertion content in prompts
3. ground-truth tables or hidden-state access
4. benchmark-specific answer mappings
5. **entity values from the scenario corpus**

**Rule 5 must distinguish schema identifiers from data values** [fixed]. The
previous wording banned "site IDs", which would have rejected every legitimate
adapter patch, since an adapter that narrows a work-order schema has to name the
field `siteid`. The line is:

- **allowed**: field and tool names from the server schemas, such as `siteid`,
  `assetnum`, `status`, `wonum`, `failurecode`
- **banned**: the values those fields take in the corpus, such as `MAIN`,
  `CHILLER6`, `1000045`, `BEARING-WEAR`, `Chiller 6 Condenser Water Flow`

Harvest the banned list from `shared/` as the distinct values of those columns,
then point `skills/tools/validate_skills.py`'s n-gram machinery at the patch.

## Phasing

**Phase 0, new.** Baseline three times over all categories. Produces the noise
floor, the per-category headroom that decides the splits, and the failure-mode
labels the proposer reads. Nothing downstream can be calibrated without it.

**Phase 1.** Journal, policy, validator, splits, loop, driven by hand-written
patches. No model. Prove the loop accepts a helping patch, reverts a regressing
one, and routes a crashing patch to debug rather than dropping it.

**Phase 2.** One run on the train categories, B=8, capability phase only, paired
dev gate. Success is a journal showing rejections and a falling regression
trend, not a headline number.

**Phase 3.** Full schedule with the steering phase, then the held-out suite
opened once.

**Phase 4.** Cross-model transfer. AutoSaddler kept +5.6 pp from Opus 4.6 to
Haiku 4.5; StarHarness kept most of its gain across GPT and Qwen.

## Cuts

Merge operators, tree search over multiple lineages, multi-epoch schedules, and
the full nine-subtype taxonomy. Start with four subtypes, all client-side:
Rule Addition, Argument Modification, Tool Description Fix, PreToolUse Hook.

## The metric that tells you it is working

Not the score. Fix rate and regression rate per iteration, AutoSaddler's
Figure 4. A working loop shows fix rates comparable to a greedy baseline and a
*falling* regression rate. If regressions climb, the dev gate is not binding and
the harness is memorizing your scenarios.

## The risk this plan does not remove

Every number in all three papers comes from benchmarks with large headroom:
GAIA2 at 53% baseline, ITBench at 40%, EnterpriseOps at 23%. If AssetOpsBench's
baseline already sits high on the categories you pick, there is little for the
search to find, and no amount of loop machinery fixes that. Phase 0 tells you
whether the premise holds before you spend Phase 2. A low-headroom result is a
finding worth reporting, not a failed setup.

## Sources

- [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/abs/2502.13138),
  [implementation](https://github.com/WecoAI/aideml)
- StarHarness, arXiv 2608.24804
- AutoSaddler, arXiv 2608.23041
