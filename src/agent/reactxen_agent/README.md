# ReActXen over Stirrup

ReActXen uses separate Actor, Review and Reflect Stirrup sessions, with one
shared Docker workspace and independent conversations. An optional Distiller
checks question clarity between the first reflection and the next actor attempt.

| Phase | Tools | Responsibility |
| --- | --- | --- |
| Actor | Domain MCP servers, code, candidate registration, optional self-ask | Collect evidence and answer the original question |
| Review | Workspace code and candidate registration; no MCP | Verify arithmetic, population semantics and original-question alignment; compare candidates |
| Reflect | Workspace code and candidate registration; no MCP | Assess alternatives and give the actor an evidence-backed correction |
| Distiller | No tools | Propose minimal wording clarification, subject to validation and equivalence audit |

Review controls whether to retry. Its verdict is not a benchmark score; the
existing evaluator scores against scenario references after the agent run.
Additional review does not guarantee improved accuracy.

## Setup

From the repository root:

```bash
uv sync
docker build -f src/agent/stirrup_agent/Dockerfile.code -t assetops-code .
```

Configure model credentials and a **dedicated benchmark CouchDB** in an env file.
The suite resets CouchDB before each scenario; do not use a shared database.
See the repository setup instructions for CouchDB, MCP data and TSFM model
prerequisites. Private scenarios stay in a separate corpus directory.

For TokenRouter, configure `TOKENROUTER_BASE_URL` and `TOKENROUTER_API_KEY`.
The actor and critics use the same `--model-id` by default. The standalone
`--critic-model` option supports a separately reported two-model configuration.

The code sandbox requires a prebuilt local image (`assetops-code`, configurable
with `STIRRUP_CODE_IMAGE`). Runtime startup never pulls an image. Docker uses
`network_mode=none`, without injected credentials or network-administration
capabilities. Code cannot search the web, make HTTP requests or install packages.
Host model API calls and actor MCP processes remain available. Review/Reflect
attach no MCP servers. Local code execution is rejected for ReActXen because it
cannot enforce this network isolation.

## Single question

Load the scenario's data into the dedicated CouchDB first, then run:

```bash
uv run --env-file /path/to/benchmark.env reactxen-agent \
  --model-id tokenrouter/MiniMax-M3 \
  --max-rounds 2 --max-turns 20 \
  --review-max-turns 20 --reflect-max-turns 20 \
  --review-finish-repair-attempts 2 --reflect-finish-repair-attempts 2 \
  --distill-question --workspace-dir /path/outside/repo/workspaces \
  --scenario-id '<id>' '<original question>'
```

Other switches:

- `--no-self-ask`: disable the optional decomposition tool.
- `--no-review`: bypass Review and Reflect.
- `--no-distill-question`: disable the question-clarification step.
- `--critic-mode completion`: use the legacy completion-only critics; this mode
  also supports `--no-code`.

The standalone round limit defaults to three; the suite defaults to two. Each
actor prompt states the remaining round budget. Its final attempt explicitly
receives no further correction cycle, while preserving earlier feedback. A final
Review may still assess that answer without creating another actor attempt.

## Full benchmark

`SCENARIO_ROOT` must contain `scenario_<id>/question.txt`, `manifest.json`, and
reference files. References are used by the evaluator, not copied into the
execution workspace. `RESULTS` must be outside the repository.

```bash
export SCENARIO_ROOT=/path/to/private/scenarios_data
export RESULTS=/path/outside/repo/reactxen-results

uv run --env-file /path/to/benchmark.env python -m benchmark.scenario_suite_runner \
  --scenario-ids all --scenario-root "$SCENARIO_ROOT" \
  --agent_name reactxen_agent --model-id tokenrouter/MiniMax-M3 \
  --reactxen-max-rounds 2 --reactxen-max-turns 20 \
  --reactxen-review-max-turns 20 --reactxen-reflect-max-turns 20 \
  --reactxen-review-finish-repair-attempts 2 \
  --reactxen-reflect-finish-repair-attempts 2 --reactxen-timeout 1200 \
  --trajectory-root "$RESULTS/trajectories" \
  --reports-root "$RESULTS/reports" \
  --stirrup-workspace-root "$RESULTS/workspaces" \
  --skip-existing --continue-on-error
```

Selectors also include `mini`, `lite`, `open`, and family selections such as
`wosr_all` or `fcc+wosr_mini`. A custom scenario-ID text file or category YAML
profile is supported. `all` means the repository's full profile, not discovery of
every folder in an arbitrary private corpus. Add `--dry-run` to inspect commands
without starting databases or model calls.

Suite ablations use `--reactxen-no-self-ask`, `--reactxen-no-review`, or
`--reactxen-no-distill-question`. Temperature and reasoning effort are shared
through the existing `--temperature` / `--reasoning-effort` options.

The timeout bounds each scenario subprocess, including its phases. Incomplete
loops produce a nonzero CLI exit; intermediate checkpoints are retained. With
`--skip-existing`, incomplete ReActXen checkpoints are retried rather than
silently treated as completed runs. Use a fresh results directory when changing
model, budgets, configuration or implementation. Keep runs sequential when they
share a CouchDB; the loader resets its data for each scenario.

The suite attaches all registered domain MCP servers, unlike a restricted
WO-only smoke configuration. Ensure the relevant model artifacts and data for
all selected families are available. Run evaluation separately with the existing
`evaluate` CLI if using `--no-evaluate`; keep workspace JSON outside the trajectory
root so artifacts are not mistaken for evaluation records.

## Shared workspace and candidate ledger

The outer loop owns container startup, preservation and cleanup. Sessions borrow
that environment; ending a phase does not erase files. A fresh run-specific host
directory is created and preserved on completion, error or cancellation.

- `mcp_results/`: complete oversized MCP snapshots, with compact handles in context.
- `context/round-NNN/actor.json`: original question, answer, full tool trace and checks.
- `context/round-NNN/evidence-index.json`: compact pointers into tools and snapshots.
- `context/round-NNN/review.json`: structured review, report and verification trace.
- `context/round-NNN/reflect.json`: validated selection and correction plan.
- `context/round-NNN/distill.json`: optional clarification proposal and equivalence audit.
- `context/round-NNN/actor-handoff.json`: exact feedback and advisory clarification.
- `candidate-ledger/index.json`: append-only candidate observations and provenance.
- `candidate-ledger/sources/`: original calculation output and tool arguments.

Code tools automatically retain labelled entity/count/value tables, scalar JSON
answers and numeric mappings. Other formats can be registered with
`retain_candidate`. Automatic extraction is not a universal arbitrary-text
parser, so agent instructions require registration of other plausible discoveries.
Retained observations are not automatically correct answers. Different filters
remain distinct even if they name the same machine; duplicate records are
provenance, not votes.

Review must compare **every retained candidate ID**, without the five-item cap
on its separate new-alternative shortlist. Reflect must assess every retained ID
and every shortlisted alternative, and name the selected supporting IDs. Missing,
unknown or duplicate assessments are rejected. Rejected/unresolved options stay
in the ledger. Original snapshots and other phases' files are preserved by prompt
instructions, not filesystem write protection.

## Verification and completion

Review returns a typed outcome, separate arithmetic/population/original-question
alignment checks, evidence and counterevidence for alternatives, and a comparison
for every retained candidate. An accomplished verdict cannot leave verification
flags false or a plausible candidate unresolved. Matching a sum is insufficient
if the selected population does not match the question. These checks constrain
the report; they do not prove the model's semantic judgments are correct.

Review and Reflect require accepted typed finish calls. Unfinished assistant
text is never promoted to an answer or correction plan. Evidence-path arrays
accept the exact unambiguous `{"item": [...]}` wrapper but reject other malformed
values. Output export paths must be individual existing files, not directories.
Each critic gets up to two reserved finish-only repair steps by default (0–3 is
configurable). Repair reuses its conversation, exposes only finish, supplies
field-specific feedback, and counts all additional usage. An incomplete phase
stays incomplete when repair fails.

The original question always remains the actor's task message and Review's
alignment target. After the first completed reflection, unresolved semantic flags
or plausible candidate interpretations can trigger Distiller. It proposes only
source-anchored wording edits; deterministic checks and a separate semantic audit
must accept a revision before an advisory clarified view is added. Failed or
rejected revisions leave the wording unchanged. Unknown domain rules remain
ambiguities, not invented requirements. Distillation draws on the original
[ReActXen Distiller](https://github.com/IBM/ReActXen/blob/main/src/reactxen/agents/distiller_agent/distiller_prompt.py)
without its generalization/Python-transformation steps.

## Traces and tests

`AGENT_TRAJECTORY_DIR` receives one outer scenario record, with nested phase
trajectories, candidate ledger, verdicts, guidance, distillation, errors, round
notices, workspace path and token usage. It is checkpointed after completed
phases. Phase records do not become competing top-level evaluation inputs.
`accomplished` is a review verdict; `max_rounds` is budget exhaustion, not approval.

```bash
uv run pytest src/agent/reactxen_agent/tests src/agent/stirrup_agent/tests \
  src/benchmark/tests/test_scenario_suite_runner.py -q
```

These are offline tests using stubs: no model credentials, CouchDB or live Docker
are needed. Compare measured accuracy, tokens, tool calls and elapsed time at
matched budgets across baseline and ablation arms. Report model, server set,
code track, sampling settings, phase limits and all failed/incomplete runs.
