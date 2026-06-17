# Contributing to AssetOpsBench

Thank you for your interest in contributing to **AssetOpsBench**! This project aims to advance the state of Industrial AI by providing a rigorous benchmarking framework for autonomous asset operations.

## How to Contribute

1. **Fork the repository** to your own GitHub account.
2. **Create a feature branch** from `main` in your fork: `git checkout -b feature/<short-topic>`.
3. **Keep PRs small and focused**: We prefer PRs with fewer than 300 changed lines to ensure high-quality reviews.
4. **Follow Conventional Commits** for all commits and PR titles.
5. **Run formatting and tests** locally before opening a pull request.
6. **Open a Pull Request** from your fork to `main` with a clear description of the benchmarking impact.

> **Note:** All PRs are merged using **Squash and merge**. The PR title will become the final commit message. Please write it carefully using the Conventional Commits format.

---

## DCO: Developer's Certificate of Origin

This repository requires a **DCO 1.1 signoff** on every commit. This is a legal statement asserting that you have the right to submit the code. You can sign off by adding the `-s` or `--signoff` flag:

```bash
git commit -s -m 'feat(eval): add predictive maintenance scoring for pumps'

```

If you have already made commits without a signoff, you can fix them:

* **Last commit only:** `git commit --amend --no-edit --signoff`
* **Multiple commits:** `git rebase --signoff HEAD~<n>` (where `<n>` is the number of commits).

Followed by a `git push -f` to your fork.

---

## Commit & Branching Standards

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

**Structure:** `<type>[optional scope]: <description>`

* `feat`: New benchmark scenario, asset model, or agentic logic (e.g., ReAct).
* `fix`: Bug fix in evaluation scripts or data loaders.
* `docs`: Documentation improvements.
* `refactor`: Code changes that neither fix a bug nor add a feature.
* `perf`: Improvements to evaluation speed or data processing.

### Branch Naming

Use the structure: `<type>/<description>`

* **Good:** `feature/hvac-chiller-scenario`, `bugfix/fix-jsonl-loader`
* **Bad:** `update1`, `feature_new_stuff` (no underscores or vague names)

---

## Local Development Setup

We use `uv` for lightning-fast Python dependency management.

### 1. Install Dependencies

```bash
uv sync --dev
source .venv/bin/activate

```

### 2. Code Quality & Formatting

We use `ruff` for both linting and formatting. Run these before every commit:

```bash
uv run ruff format .
uv run ruff check --fix .

```

### 3. Security Scanning

To protect industrial metadata and API keys, run the IBM `detect-secrets` scan:

```bash
uv pip install --upgrade "git+[https://github.com/ibm/detect-secrets.git@master#egg=detect-secrets](https://github.com/ibm/detect-secrets.git@master#egg=detect-secrets)"
detect-secrets scan --update .secrets.baseline
detect-secrets audit .secrets.baseline

```

---

## Running Tests & Validation

### Unit Tests

Validate core logic for metrics and data parsing:

```bash
uv run pytest tests/unit

```

### Integration & Benchmark Validation

Verify that agent trajectories and environment simulations run correctly:

```bash
chmod +x ./scripts/run_tests.sh
./scripts/run_tests.sh

```

This script validates:

* **Linting**: Ruff validation.
* **Agentic Logic**: Verification of ReAct and Plan-Execute orchestration.
* **Asset Consistency**: Ensuring industrial asset IDs (e.g., FailureSensorIQ) match registry definitions.

---

## Pull Request Guidelines

* **Benchmark Integrity**: If your change modifies existing scoring logic, please include a "Before vs. After" comparison in the PR description.
* **Asset Privacy**: Ensure no real-world sensitive telemetry data is included in scenarios without anonymization.
* **Documentation**: Update the relevant dataset cards (e.g., for FailureSensorIQ) if you modify the underlying data structures.
* **PR Templates**: Use the provided templates for Features, Bug Fixes, or Chores to ensure consistent review cycles.

