# AssetOpsBench Contributor Execution Prompt & TODO

Repository: https://github.com/IBM/AssetOpsBench

## Purpose

Use this file as the execution plan for becoming a useful, visible contributor to AssetOpsBench through small, high-quality, maintainable contributions.

The goal is **not** to spam maintainers or open five PRs at once. The goal is to build credibility by fixing real problems, communicating clearly, respecting existing assignees, and keeping each PR focused.

---

# MASTER PROMPT FOR CHATGPT / CODEX

You are acting as my senior open-source contributor and coding agent.

Your target repository is:

https://github.com/IBM/AssetOpsBench

Your mission is to help me become a recognized, useful contributor to this project by executing the five contribution opportunities below carefully and professionally.

## Core rules

1. Work from the latest `main`. Before starting any contribution:
   - fetch upstream,
   - inspect the latest commits,
   - inspect current open issues,
   - inspect current open PRs,
   - verify the problem still exists,
   - do not rely on stale assumptions.

2. Never push directly to `IBM/AssetOpsBench/main`.
   - Use my fork.
   - Configure:
     - `upstream` = `https://github.com/IBM/AssetOpsBench.git`
     - `origin` = my fork.
   - If my fork does not exist, ask me to create/connect it or use the available GitHub tooling to create it if permitted.

3. Keep PRs small and focused.
   - Prefer fewer than 300 changed lines.
   - One logical concern per PR.
   - Do not bundle unrelated fixes.

4. Follow repository conventions:
   - Python 3.12+
   - `uv`
   - Conventional Commit titles
   - DCO sign-off on every commit using `git commit -s`
   - English only
   - Run relevant tests before pushing.

5. Do not create fake work just for visibility.
   - Do not add unnecessary files to upstream.
   - Do not add my personal TODO file to an upstream PR unless maintainers explicitly ask.
   - Visibility should come from useful issue comments, reproducible findings, clean PRs, tests, and thoughtful design notes.

6. Before opening a PR, search for:
   - an existing issue,
   - an existing PR,
   - an active maintainer branch,
   - an assignee already working on the same task.

7. If an issue is already assigned:
   - do not race the assignee,
   - leave a concise comment offering help,
   - only implement after there is a reasonable indication that collaboration is welcome.

8. After every completed contribution:
   - update this TODO file in my fork,
   - record branch name,
   - commit SHA,
   - PR URL,
   - tests run,
   - current status,
   - next action.

---

# INITIAL SETUP

Clone my fork and configure upstream.

Example:

```bash
git clone <MY_FORK_URL>
cd AssetOpsBench
git remote add upstream https://github.com/IBM/AssetOpsBench.git
git fetch upstream
git checkout main
git reset --hard upstream/main
git push origin main
```

Create a personal tracking branch:

```bash
git checkout -b contrib/assetopsbench-roadmap
```

Create or update this file as:

```text
CONTRIBUTOR_TODO.md
```

Commit and push it to **my fork only**:

```bash
git add CONTRIBUTOR_TODO.md
git commit -s -m "chore: add AssetOpsBench contributor roadmap"
git push -u origin contrib/assetopsbench-roadmap
```

Do **not** open an upstream PR for this personal roadmap.

After that, start Contribution 1.

---

# CONTRIBUTION 1 — Fix contributor onboarding and test instructions

## Goal

Verify whether contributor documentation still contains stale or broken development/test commands.

Previously observed examples included references such as:

```bash
uv run pytest tests/unit
uv run pytest tests/integration
./scripts/run_tests.sh
```

while the repository used tests under `src/`.

## Required investigation

Inspect at least:

- `CONTRIBUTING.md`
- `INSTRUCTIONS.md`
- `.github/PULL_REQUEST_TEMPLATE/bugfix.md`
- `.github/PULL_REQUEST_TEMPLATE/feature.md`
- `pyproject.toml`
- test directories under `src/`
- current scripts under `scripts/`

Also verify whether Ruff is actually declared as a development dependency.

## If the problem still exists

Create a dedicated branch such as:

```bash
git checkout main
git reset --hard upstream/main
git checkout -b docs/fix-contributor-test-instructions
```

Make the smallest correct patch.

Possible scope:

- replace invalid pytest paths with commands that match the current repository;
- remove references to scripts that no longer exist;
- make `CONTRIBUTING.md` consistent with `INSTRUCTIONS.md`;
- if Ruff is required but not installed by the documented setup, either:
  - add Ruff to the appropriate development dependency group, or
  - document the actual installation path;
- update the bugfix PR template if it contains a nonexistent test command.

## Verification

Run the exact commands you document.

At minimum, where applicable:

```bash
uv sync
uv run pytest src/ -k "not integration"
```

If Ruff is part of the patch:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Suggested PR title

```text
docs(contributing): align test and development instructions
```

If dependency metadata must also change substantially, split that into Contribution 2 rather than bloating this PR.

## PR quality

In the PR description include:

- what was broken;
- exact stale commands;
- exact replacement commands;
- how you verified them;
- confirmation that this changes contributor experience only and not benchmark behavior.

---

# CONTRIBUTION 2 — Clean project metadata and duplicate dependencies

## Goal

Audit `pyproject.toml` for obvious packaging metadata problems and duplicate dependency declarations.

Previously observed candidates included:

- placeholder project description;
- duplicate `scipy`;
- duplicate/conflicting `sktime` minimum versions.

## Required investigation

Do not assume these still exist.

Check the latest:

```text
pyproject.toml
uv.lock
```

Search issues/PRs for related work before editing.

## If still valid

Create a separate branch:

```bash
git checkout main
git reset --hard upstream/main
git checkout -b chore/clean-project-metadata
```

Make a minimal cleanup:

- replace placeholder metadata with an accurate description based on the README/project purpose;
- remove exact duplicate requirements;
- consolidate duplicate package constraints to the effective intended minimum version;
- do not casually upgrade unrelated dependencies.

Regenerate the lockfile only if necessary:

```bash
uv lock
```

## Verification

```bash
uv sync
uv run pytest src/ -k "not integration"
```

## Suggested PR title

```text
chore(packaging): clean project metadata and duplicate dependencies
```

Explain why dependency resolution behavior is unchanged if that is true.

---

# CONTRIBUTION 3 — Add service-free PR test CI

## Goal

Determine whether normal pull requests currently run the Python unit test suite.

Previously the workflows appeared focused on security/data guards rather than project tests.

## First step: discuss before coding

Because CI affects every contributor, first inspect:

```text
.github/workflows/
```

Search open and closed issues/PRs for CI discussions.

If there is no active equivalent work, open a concise issue or discussion proposing:

- a lightweight Python 3.12 job;
- `uv` dependency installation;
- service-free tests only;
- no CouchDB/WatsonX requirement;
- caching if appropriate;
- Ruff only if the project officially supports it.

Suggested core test command:

```bash
uv run pytest src/ -k "not integration"
```

Ask maintainers whether they prefer:
- only tests;
- tests + Ruff;
- one Python version or a version matrix.

## Only implement after checking maintainer direction

Suggested branch:

```bash
ci/add-unit-test-workflow
```

Suggested PR title:

```text
ci: run service-free tests on pull requests
```

Keep the workflow minimal.

Do not add expensive integration services unless requested.

---

# CONTRIBUTION 4 — Help with FMEA Lite scenarios (#582)

## Goal

Investigate issue:

https://github.com/IBM/AssetOpsBench/issues/582

Previously the Lite scenario profile had an empty FMEA list while the full and mini profiles contained FMEA scenarios.

## Important etiquette

This issue may already be assigned.

Before changing anything:

1. Read the current issue.
2. Inspect assignees.
3. Search current PRs.
4. Inspect:
   - `benchmarks/scenario_suite/lite.yaml`
   - `benchmarks/scenario_suite/mini.yaml`
   - `benchmarks/scenario_suite/all.yaml`
   - `benchmarks/scenario_suite/README.md`
   - relevant FMEA evaluator tests.

If the issue is still assigned with no PR, leave a short comment similar to:

> Hi, I’m starting to contribute to AssetOpsBench and I’d be happy to help with #582. I see the issue is assigned, so I don’t want to duplicate work. If useful, I can prepare the Lite FMEA selection and validation/tests. Let me know if you’d like me to take it or collaborate.

Do not open a competing PR until collaboration is appropriate.

## If invited to implement

Choose five representative FMEA scenarios based on documented suite design, not randomly.

Add/adjust tests if the suite loader has profile validation.

Suggested PR title

```text
feat(benchmark): add FMEA scenarios to lite profile
```

Explain the selection criteria.

---

# CONTRIBUTION 5 — Participate in FMEA design (#550)

## Goal

Contribute meaningfully to:

https://github.com/IBM/AssetOpsBench/issues/550

This is a design problem around representing richer FMEA concepts such as:

- failure mode;
- mechanism/cause;
- effect;
- potentially severity, occurrence, detection, controls, or other FMEA fields.

## Do not code first

Start with investigation.

Read:

- the current FMEA issue and comments;
- FMEA scenario ground truth;
- FMEA evaluator/scorer implementation;
- FMEA documentation;
- benchmark scenario schema;
- any related scenario metadata;
- existing tests.

Then prepare a concise design proposal in the issue.

The proposal should answer:

1. What is the current FMEA representation?
2. What information is missing?
3. Where should new fields live?
4. Which fields should be required vs optional?
5. How should backward compatibility work?
6. How should evaluation score partial correctness?
7. How will old FMEA scenarios continue to work?
8. What is the smallest first implementation PR?

Prefer a phased proposal such as:

- Phase 1: schema/ground-truth extension;
- Phase 2: scorer support;
- Phase 3: scenario updates;
- Phase 4: benchmark/reporting enhancements.

Do not create a giant PR before maintainers agree on the design.

---

# VISIBILITY / MAINTAINER ATTENTION STRATEGY

We want maintainers to notice my work for the right reasons.

Use these methods:

1. Reproduce before fixing.
   - Give exact file/command evidence.

2. Leave concise, useful issue comments.
   - State what you checked.
   - Avoid generic “Can I work on this?” comments when you can provide analysis.

3. Keep PRs easy to review.
   - Small diff.
   - Good commit title.
   - Clear tests.
   - No unrelated formatting churn.

4. Link issues correctly.
   - Use `Fixes #...` only when the PR genuinely resolves the issue.
   - Otherwise use `Related to #...`.

5. Respect assigned work.
   - Collaboration gets more positive attention than racing maintainers.

6. Report pre-existing failures separately.
   - Do not hide them.
   - Prove whether they also fail on upstream `main`.

7. Use DCO sign-off on every commit:

```bash
git commit -s -m "..."
```

8. After merge, thank reviewers briefly and move to the next item.

9. Do not open all five PRs at once.
   - Finish or stabilize one before opening the next.

10. Prefer one excellent contribution over five rushed ones.

---

# EXECUTION TODO

## Repository setup

- [x] Confirm my GitHub fork exists.
- [x] Clone my fork.
- [x] Add IBM repository as `upstream`.
- [x] Fetch and sync latest `upstream/main`.
- [x] Create `contrib/assetopsbench-roadmap`.
- [x] Add this `CONTRIBUTOR_TODO.md`.
- [x] Commit with DCO sign-off.
- [x] Push roadmap branch to my fork only.

## Contribution 1 — Contributor onboarding

- [x] Re-check latest `CONTRIBUTING.md`.
- [x] Re-check latest `INSTRUCTIONS.md`.
- [x] Re-check PR templates.
- [x] Verify actual test layout.
- [x] Verify whether `scripts/run_tests.sh` exists.
- [x] Verify whether Ruff is installed through project dependencies.
- [x] Search existing issues/PRs for duplicate work.
- [x] Create focused branch.
- [x] Implement minimal fix.
- [x] Run documented commands.
- [x] Commit with DCO sign-off.
- [x] Push branch.
- [x] Open PR (#628; submitted manually).
- [x] Record PR URL here.
- [x] Address review (two approvals, no changes requested).
- [x] Record merge status.

Status: MERGED — CONTRIBUTION 1 COMPLETED  
Branch: docs/fix-contributor-test-instructions  
Original commit SHA: 8dae067cbca4bcca46ff6a910623e32d5fba9ef9  
Upstream merge commit: 6665eab8ce210eb15852f0d33ede1e24dd1a08fc  
PR: https://github.com/IBM/AssetOpsBench/pull/628  
Notes: Merged October 11, 2026, at 21:31:05 UTC by DhavalRepo18 after two APPROVED reviews, with no requested code changes. Three contributor documentation files incorporated in upstream main. DCO, TruffleHog and Gitleaks checks succeeded. Original local tests: 33 focused tests passed; broader service-free command was interrupted with exit 137 in the available environment. No further work required for this PR. Contribution 2 packaging preflight is investigated separately; create a new focused branch only when ready to implement.  

## Contribution 2 — Packaging cleanup

- [x] Re-check latest `pyproject.toml`.
- [x] Verify duplicate/placeholder issues still exist.
- [x] Search for competing PRs.
- [x] Create focused branch.
- [x] Make minimal metadata/dependency cleanup.
- [x] Run `uv lock` only if required.
- [x] Run tests.
- [x] Commit with DCO sign-off.
- [x] Push branch.
- [x] Open PR (#629).
- [x] Record PR URL here.
- [x] Address review (approved; no changes requested).

Status: MERGED — CONTRIBUTION 2 COMPLETED  
Branch: chore/clean-project-metadata  
Original commit: a88959d5b558c63f00fef3e3a773af021e4f10e9  
Upstream merge commit: e844f6468c6c68b328022e7ca8e31da140842cc7  
PR: https://github.com/IBM/AssetOpsBench/pull/629  
Notes: Approved by DhavalRepo18 on October 13, 2026 at 02:10:29 UTC, merged by DhavalRepo18 at 02:10:37 UTC without requested changes. One file (pyproject.toml), 2 additions and 4 deletions: description update, remove duplicate scipy, retain effective sktime>=1.0.1. DCO, Gitleaks and TruffleHog all passed on the PR commit. Prior local uv lock --check, uv lock (284 packages), uv sync --frozen, git diff --check, and 33 focused unit tests all passed. The TSFM test selection was not completed (runner timeout). No uv.lock changes were required. Fork main synchronized to upstream merge commit. No further action needed on this PR; Contribution 3 can be assessed separately.  

## Contribution 3 — PR CI

- [x] Inspect current workflows.
- [x] Search CI-related issues/PRs.
- [x] Draft minimal CI proposal.
- [x] Open issue/discussion if needed.
- [x] Get maintainer direction (`Try one`, 2026-10-14).
- [x] Implement focused first version in fork only.
- [x] Validate workflow in fork Actions.
- [ ] Open focused upstream PR (connector attempt blocked by 403; use authenticated GitHub surface).
- [x] Record branch/SHA/validation here.

Status: IMPLEMENTED AND FORK-VALIDATED — UPSTREAM PR NOT YET CREATED  
Issue/Discussion: https://github.com/IBM/AssetOpsBench/issues/632 (open; assigned to yousseffaouzi)  
Proposal/status: https://github.com/yousseffaouzi/AssetOpsBench/blob/contrib/assetopsbench-roadmap/docs/CONTRIBUTION3_PR_CI_PROPOSAL.md  
Branch: `ci/add-python-smoke-workflow`  
Implementation commit: `cd7b97da1affed26a1694708d59df23141eaebc0`  
Fork validation PR: https://github.com/yousseffaouzi/AssetOpsBench/pull/3 (draft, validation only)  
Fork workflow run: `38656000694` — SUCCESS  
Upstream PR: Not created; connected integration returned HTTP 403 `Resource not accessible by integration`.  
Notes: Preflight on 2026-10-15 verified upstream and fork main still aligned at `e844f6468c6c68b328022e7ca8e31da140842cc7`, only the three known upstream workflows were present, and no competing open Python smoke-workflow PR was found. The implementation adds only `.github/workflows/python-smoke.yml` (65 lines), with `permissions: contents: read`, path filters, PR concurrency cancellation, 10-minute timeout, `actions/checkout@v7`, official shell-installed uv, `uv lock --check`, frozen-lock constraints, and only the direct dependencies needed by the targeted service-independent tests. Fork Actions confirmed checkout, uv install, lock check, constrained dependency install, and `.venv/bin/python -m pytest src/llm/tests src/servers/utilities/tests -q` all succeeded. Local preparation parsed the YAML and verified required uv CLI flags, but the local shell could not resolve GitHub/PyPI hosts, so no local pytest pass is claimed. No benchmark scoring/data, external services, credentials, TSFM-heavy suites, Ruff changes, branch-protection changes, or mandatory-gate changes were made. Next: open a draft upstream PR from the fork branch using an authenticated GitHub surface with permission, link it as `Related to #632` rather than `Fixes`, and observe IBM Actions-policy behavior before any further change.  

## Contribution 4 — FMEA Lite scenarios (#582)

- [ ] Read current #582.
- [ ] Check assignee/current activity.
- [ ] Search for open PRs.
- [ ] Inspect Lite/Mini/All profiles.
- [ ] Inspect FMEA scenario/test coverage.
- [ ] Leave useful collaboration comment if still assigned.
- [ ] Wait for/observe maintainer direction before competing implementation.
- [ ] If invited, select five representative scenarios.
- [ ] Add validation/tests if appropriate.
- [ ] Open focused PR.
- [ ] Record PR URL here.

Status: NOT STARTED  
Issue: https://github.com/IBM/AssetOpsBench/issues/582  
Branch:  
PR:  
Notes:  

## Contribution 5 — FMEA design (#550)

- [ ] Read current #550.
- [ ] Inspect FMEA schema/ground truth.
- [ ] Inspect scorer/evaluator.
- [ ] Inspect docs/tests.
- [ ] Identify missing FMEA concepts.
- [ ] Draft backward-compatible design.
- [ ] Propose phased implementation.
- [ ] Post design comment to #550.
- [ ] Wait for maintainers to align on direction.
- [ ] Implement only the agreed first phase.
- [ ] Open focused PR if requested/agreed.

Status: NOT STARTED  
Issue: https://github.com/IBM/AssetOpsBench/issues/550  
Design comment:  
PR:  
Notes:  

---

# AGENT OPERATING LOOP

For every contribution, follow this loop:

```text
SYNC
→ VERIFY PROBLEM
→ SEARCH DUPLICATE WORK
→ DISCUSS IF NEEDED
→ CREATE SMALL BRANCH
→ IMPLEMENT
→ TEST
→ SELF-REVIEW
→ DCO COMMIT
→ PUSH
→ OPEN CLEAN PR
→ RESPOND TO REVIEW
→ UPDATE CONTRIBUTOR_TODO.md IN MY FORK
→ NEXT CONTRIBUTION
```

At every step, show me:
- what you found;
- what files you changed;
- why;
- tests run and results;
- any pre-existing failures;
- Git branch;
- commit SHA;
- PR/issue URL;
- recommended next action.

Do not silently make broad architectural changes.

Start now with repository setup and Contribution 1.