# Adaptive escalation: evidence-driven live validation

Status: **exploratory engineering validation, not a final benchmark claim**

Date: 2026-09-01

## Executive conclusion

The current redesign is a useful project improvement, but for narrower reasons than
the original proposal implied.

It does not yet have a statistically established task-accuracy advantage. In a final
single redesigned run it answered the two locally answerable count scenarios exactly
and failed the unsupported end-to-end scenario, for 2/3 strict passes. A current
baseline/redesigned pair also passed scenario 2, so that accuracy result cannot be
attributed solely to adaptive escalation.

The demonstrated adaptive improvement is evidence integrity and bounded work:

- successful specialist-server runs are no longer verified merely because they are
  deep or contain domain vocabulary;
- an unadvertised argument was rejected and repaired before any tool call;
- post-call retries occurred only for explicitly read-only tools;
- a failed prerequisite stopped four dependent steps;
- the runner returned an explicit failure instead of fabricated anomaly output; and
- on the matched current scenario-3 check, redesigned execution used 16,536 tokens
  and four LLM calls versus baseline's 22,863 tokens and seven calls, a 27.7% token
  reduction in that diagnostic pair; the redesigned trajectory stopped unsupported
  downstream work.

This is useful behavior for AssetOpsBench even when the task cannot be completed. It
is safer and more auditable than critique-only verification. It should not be sold as
an accuracy win until a larger preregistered repeated study supports that claim.

## Questions evaluated

1. Does the policy avoid escalation on successful work that merely looks specialized?
2. Can recovery correct a concrete execution defect without replaying unsafe tools?
3. Does the runner stop when required evidence becomes unavailable?
4. Does it avoid claiming successful completion from failed tool evidence?
5. Do live evaluator results survive inspection against raw ground truth?
6. Is any observed outcome large and controlled enough to call an improvement?

## Conditions

Within each comparison, live runs used the same WatsonX model at temperature 0, a
CouchDB reset before every condition, the same local scenario fixtures, and the
`static_json` evaluator. The diagnostic matrix and final follow-up use different code
checkpoints and are reported separately; their scores are not combined.

| Condition | Execution behavior | Routing behavior |
| --- | --- | --- |
| Baseline | Legacy execution | No verification |
| Original | Legacy execution | PR #432 broad critique-only policy |
| Redesigned | Strict error detection and bounded safe recovery | Direct execution evidence |
| Always | Redesigned execution | Verification forced on |

Each recorded condition was run once. Provider calls were not seeded or replayed, so
plans varied. Comparisons are diagnostic associations, not causal effect estimates.

The experiment harness now requires `--acknowledge-external-llm`, supports a selected
condition subset, records the decision made by the actual condition, and labels
incomplete matrices as mechanism checks. Repository scenario text and tool evidence
were sent to WatsonX only after explicit approval; `.env` values were not included.

## Design boundary

The redesign acts on execution evidence rather than guessing risk from vocabulary or
plan shape:

```text
step succeeds                         -> continue
arguments fail before a tool call     -> regenerate once within the run budget
read-only tool call fails             -> regenerate and retry once
mutating or unknown-safety call fails -> do not replay; verify the failure
required evidence remains unavailable -> block dependants and report failure
all steps succeed                     -> summarize normally
```

Adaptive behavior remains opt-in. Recovery is bounded to one retry per step and a
small run-wide budget. A post-call retry requires MCP metadata that explicitly marks
the tool read-only and non-destructive; idempotence alone is not enough. A failed
dependency is never called with placeholder evidence. `retry_step` records that a
bounded recovery already succeeded—it does not initiate another tool call after the
plan finishes.

## Evaluation integrity correction

The first live matrix exposed a false-positive scorer result. Scenario 2 has scalar
ground truth `1`, but the answer "The final count cannot be provided due to the failure
in Step 1" was scored as an exact match because the scalar parser extracted the only
number in the prose.

The scorer now accepts a scalar count only when it is:

- count-only;
- explicitly labelled as the answer, count, or result; or
- the final standalone numeric line.

Two regression tests cover the failed-step phrase and unrelated numeric prose. After
offline rescoring, the original-policy scenario-2 result correctly changed from pass
to failure. The pre-correction pass is not used anywhere in this report.

## Diagnostic four-condition matrix

This matrix preceded the follow-up planner/argument fixes and was used to find design
failures rather than support a positive claim.

| Condition | Strict passes | Mean score | Tokens | LLM calls | Verify rate | Recovery rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1/3 | 0.381 | 47,001 | 14 | 0% | 0% |
| Original | 0/3 | 0.048 | 50,393 | 16 | 100% | 0% |
| Redesigned, first version | 1/3 | 0.333 | 39,802 | 12 | 0% | 67% |
| Always | 1/3 | 0.381 | 42,708 | 13 | 100% | 33% |

This result did **not** show a reliability improvement. It did show:

- the original policy still verified every case;
- expensive verification did not repair wrong evidence;
- one recovery returned operational success with semantically wrong filters and an
  empty result; and
- structured tool errors remained invisible in legacy execution.

The work therefore continued locally instead of publishing the matrix as a success.

## Evidence-driven follow-up changes

The diagnostic failures led to small general changes rather than scenario-specific
rules:

1. The planner prompt instructs the model to resolve indirectly described identifiers
   using an available discovery/list tool before the consuming step.
2. The planner prompt forbids planning any capability absent from the advertised tool
   set. This remains prompt-level guidance, so runtime capability and dependency checks
   remain necessary.
3. Argument resolution receives the tool description separately from its parameter
   names.
4. Optional filters are omitted when the question asks for all values and are included
   only when an exact value is supported by the question, tool documentation, or prior
   evidence.
5. Unadvertised parameter names fail validation before a tool call and are eligible
   for one safe repair.
6. Verification and summarization evidence includes the exact tool name and arguments.

These constraints apply across tools and scenarios. They do not encode expected
answers, scenario IDs, work-order numbers, or ground-truth fields.

## Final current redesigned run

| Scenario | Ground truth | Answer/action | Strict | Recovery | Tokens | Calls |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `2` | `2`; `none` | 1.0 | No | 13,686 | 4 |
| 2 | `1` | `1`; `retry_step` | 1.0 | Yes | 14,611 | 5 |
| 3 | Structured anomaly result | Explicit `report_failure` | 0.0 | Attempted, exhausted | 16,536 | 4 |

Aggregate: 2/3 strict passes, mean score 0.667, 44,833 tokens, 13 LLM
calls, estimated cost $0.01299.

Independent rescoring parsed the final answers as integer `2`, integer `1`, and an
explicit failure string respectively, matching the harness's 1.0, 1.0, and 0.0 strict
scores.

### Scenario 1: selectivity

The run used the specialist `wo` server, dependency depth 3, and matched `work order`,
`failure`, and `anomaly`. Those signals would have triggered the original policy. All
steps succeeded, so the redesigned policy selected `none` and returned the exact count.
This is direct selectivity evidence.

### Scenario 2: bounded pre-call recovery

The argument model invented `filter` for the no-argument `sites` tool. Strict schema
validation rejected it before execution. One `safe_pre_call` repair produced `{}`;
site discovery returned `MAIN`, the work-order query used documented arguments, and
the final answer matched ground truth.

This demonstrates the intended recovery mechanism within one trajectory. It does not
prove an aggregate accuracy advantage: a separate current baseline run also answered
scenario 2 correctly after the shared planning improvements.

### Scenario 3: evidence integrity

The current baseline and redesigned conditions both failed the task. Their behavior
was materially different:

| Measure | Current baseline | Current redesigned |
| --- | ---: | ---: |
| Strict score | 0.0 | 0.0 |
| Tool failures recorded | 0 | 1 root failure plus 4 blocked dependants |
| Downstream tool calls after missing prerequisite | 3 | 0 |
| LLM calls | 7 | 4 |
| Tokens | 22,863 | 16,536 |

Baseline received structured errors from asset lookup, history retrieval, file read,
and anomaly execution but treated them as successful evidence. Its answer explicitly
said it was simulating hypothetical success and emitted invented structured values,
including zero observations and `anomalies_found: false`.

Redesigned execution detected the first structured lookup error, made one read-only
retry, exhausted that retry, blocked every dependant, and returned a deterministic
failure containing the exact missing evidence. This is an improvement in grounding,
safety, and bounded cost, not task accuracy.

## What is and is not established

Established by deterministic tests and inspected live trajectories:

- broad static vocabulary no longer routes successful work;
- strict schema validation can prevent a malformed call;
- safe recovery is bounded;
- mutating and unknown-safety failures are not replayed;
- structured errors and failed dependencies are visible;
- dependent execution stops after missing evidence; and
- failure answers do not claim completion.

Not established:

- a statistically reliable accuracy improvement;
- stability across repeated plans, other models, or the closed scenario corpus;
- that recovery's operational success implies semantic correctness; or
- that scenario 3 can be solved with the currently exposed tool graph.

## Recommendation

The adaptive work is useful enough to retain as one coherent PR, organized into
reviewable commits:

1. Production behavior and focused tests: direct-evidence routing, typed actions,
   strict structured-error handling, dependency blocking, bounded recovery, tool
   safety annotations, and deterministic failure reporting.
2. Evaluation rigor and tests: the scalar false-positive correction and controlled
   four-condition harness.
3. This evidence report.

Do not publish a claim that adaptive escalation improves benchmark accuracy from these
runs. A defensible claim is that it reduces indiscriminate verification and prevents
failed evidence from being treated as completed work.

## Reproduction and accounting

The live harness can be rerun, after starting CouchDB and explicitly authorizing the
external model calls, with:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/adaptive_escalation_experiment.py \
  --scenario-root src/couchdb/scenarios_data \
  --scenario-ids 1,2,3 \
  --acknowledge-external-llm \
  --output-dir /tmp/adaptive-escalation-results
```

The output directory must be empty. The harness resets and loads each scenario before
each condition, checkpoints partial results, writes JSON and CSV summaries, and records
the exact model and per-call token usage. Raw trajectories are intentionally excluded
from this PR; the inspected facts needed to audit the conclusions are stated above.

Measured WatsonX use, including the initial 85-token quota smoke, was 288,928 tokens.
Estimated total cost was approximately $0.0843. No further provider calls were made
after the final baseline scenario-3 check.
