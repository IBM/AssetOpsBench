# Contribution 3 — PR CI proposal and implementation status

## Implementation status (verified 2026-10-15)

- Upstream issue #632 remains open and assigned to `yousseffaouzi`.
- Maintainer `DhavalRepo18` replied on 2026-10-14: `@yousseffaouzi, we had those but did not operationalize them. Try one.` This cleared the earlier waiting state enough to prototype a focused first version.
- Upstream `IBM/AssetOpsBench/main` and fork `yousseffaouzi/AssetOpsBench/main` were both verified at `e844f6468c6c68b328022e7ca8e31da140842cc7` before implementation.
- Current upstream workflows remain `secret-scan.yml`, `guard-couchdb-data.yml`, and `stale.yml`; no competing open Python smoke-workflow PR was found in the preflight search.
- Implementation branch: `ci/add-python-smoke-workflow` in the contributor fork only.
- Implementation commit: `cd7b97da1affed26a1694708d59df23141eaebc0` (`ci: add service-free Python smoke workflow`), DCO-signed.
- Diff: one added file, `.github/workflows/python-smoke.yml`, 65 additions; no benchmark, scoring, data, Ruff, or roadmap file is present on the implementation branch.
- Fork-only validation PR: `yousseffaouzi/AssetOpsBench#3` (draft).
- Fork workflow run `38656000694` completed successfully. Checkout, uv install, `uv lock --check`, constrained smoke dependency install, and `.venv/bin/python -m pytest src/llm/tests src/servers/utilities/tests -q` all completed successfully.
- Local preparation validated YAML parsing/structure and the required uv CLI flags. The local shell could not resolve GitHub/PyPI hosts, so no local dependency install or pytest pass is claimed.
- Attempting to create a draft upstream PR through the connected GitHub integration returned HTTP 403 `Resource not accessible by integration`; no upstream PR was created through that integration.
- IBM organization compatibility remains an upstream-policy question. The prototype introduces no new third-party GitHub Action: it uses the already-present `actions/checkout@v7` and installs uv with the official shell installer.

Next action: submit the focused draft PR from `yousseffaouzi:ci/add-python-smoke-workflow` to `IBM/AssetOpsBench:main` using an authenticated GitHub surface that has permission, reference #632 as related work (do not use `Fixes`), then observe IBM Actions-policy behavior and maintainer feedback. Do not make the check mandatory or change branch protection without explicit maintainer direction.

## Current state (verified on IBM/AssetOpsBench main, 2026-10-13)

- `.github/workflows/secret-scan.yml` runs Gitleaks and TruffleHog on PRs.
- `.github/workflows/guard-couchdb-data.yml` checks newly added data files only for `src/couchdb/**` changes.
- `.github/workflows/stale.yml` is scheduled/on-demand; it is not a PR test gate.
- There is no PR-triggered general Python test workflow.
- `CONTRIBUTING.md` documents `uv run pytest src/ -k "not integration"`, while `INSTRUCTIONS.md` explains integration prerequisites.
- Existing workflow comments note IBM's restriction on third-party Actions. Existing CI uses `actions/checkout@v7` and shell-installed utilities.

## Suggested upstream issue

**Title:** ci: discuss a lightweight Python smoke check for pull requests

**Body:**

### Problem

AssetOpsBench has PR-triggered secret scanning and a CouchDB-data guard, but no general Python test gate for changes under `src/` or dependency metadata. Recent documentation/packaging PRs (#628 and #629) passed DCO and secret checks, while Python tests remained manual.

The workflows explicitly accommodate IBM's organization Actions policy by relying on an approved checkout Action and shell commands. A new CI job should respect this constraint.

### Proposed small first step (seeking feedback, not implementing yet)

1. Add one PR-triggered, read-only workflow for Python source and dependency/configuration changes, with reasonable job timeout and concurrency.
2. Use `permissions: contents: read` and the approved checkout action; install Python/uv through an organization-compatible method, avoiding unapproved third-party Actions.
3. Run `uv lock --check` for metadata/lock consistency.
4. Run a small service-independent Python smoke selection, such as `uv run --frozen pytest src/llm/tests src/servers/utilities/tests -q` (33 tests previously passed locally).
5. Exclude heavy TSFM, agent integration, CouchDB and external LLM-backed suites from the initial check.
6. Do not make repository-wide Ruff checks mandatory while pre-existing unrelated lint/format violations remain.

No benchmark scoring, agent behavior, scenarios, or data change is requested.

### Questions for maintainers

- Is a small Python smoke check desirable? Informational initially or required for merges?
- `uv run` can resolve/install heavy runtime/ML dependencies even for a tiny test selection. Prefer an isolated minimal test environment or a bounded full install?
- Which GitHub Actions setup/install methods are approved under IBM's policy?
- Suggested path filters: `src/**`, `pyproject.toml`, `uv.lock`, and the new workflow. Are other paths needed?

I have not implemented the workflow or opened a PR, pending maintainer input.

Relevant files: `.github/workflows/secret-scan.yml`, `.github/workflows/guard-couchdb-data.yml`, and `INSTRUCTIONS.md#running-tests`.

## Contribution constraints

- No upstream CI change before maintainer response, as required by `CONTRIBUTOR_TODO.md`.
- Minimal separate branch from latest upstream main after scope is confirmed.
- Run actual workflow-equivalent checks before proposing a PR.
- No proprietary credentials, external services, or benchmark changes.

## Submission status

Attempted upstream GitHub issue creation through the connected GitHub integration on 2026-10-13; returned `403 Resource not accessible by integration`. Submit via authenticated GitHub browser to IBM/AssetOpsBench Issues and record the issue number/URL in the roadmap.
