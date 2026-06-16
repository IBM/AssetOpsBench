# Blocking Secrets in IBM/AssetOpsBench — Setup Guide

A layered ("defense in depth") setup so sensitive values (API keys, tokens,
passwords, private keys) are caught at three points: the developer's machine,
the CI pipeline, and GitHub's own push gateway.

## File placement

| File in this folder        | Put it at repo path             |
|----------------------------|---------------------------------|
| `pre-commit-config.yaml`   | `.pre-commit-config.yaml` (rename) |
| `.gitleaks.toml`           | `.gitleaks.toml`                |
| `secret-scan.yml`          | `.github/workflows/secret-scan.yml` |

---

## Layer 1 — Local pre-commit hook (catches secrets before a commit exists)

```bash
# one-time, per contributor
pip install pre-commit detect-secrets

# generate the baseline of currently-known/allowed values
detect-secrets scan > .secrets.baseline

# install the git hook into this clone
pre-commit install

# (optional) test against the whole repo right now
pre-commit run --all-files
```

After this, every `git commit` runs gitleaks + detect-secrets on staged changes
and aborts the commit if a secret is found. Commit `.pre-commit-config.yaml`,
`.gitleaks.toml`, and `.secrets.baseline` to the repo so the whole team shares
the config. 

---

## Layer 2 — GitHub Actions CI check (server-side, blocks PR merges)

`secret-scan.yml` runs gitleaks **and** TruffleHog on every push and pull
request, plus a weekly full-history sweep. Nothing to install — just commit the
workflow file. To make it enforce merges:

1. Repo **Settings → Branches → Add branch protection rule** for `main`.
2. Enable **Require status checks to pass before merging**.
3. Select the **Gitleaks** and **TruffleHog** checks.

Now a PR that introduces a secret cannot be merged until it's removed.

---

## Layer 3 — GitHub native Secret Scanning + Push Protection (no code)

This is GitHub's built-in gateway that rejects a `git push` the moment it
detects a recognized secret pattern.

1. Go to the repo on GitHub → **Settings → Code security and analysis**
   (org-level admins may set this for all repos under the IBM org).
2. Enable **Secret scanning**.
3. Enable **Push protection**.

For private/internal repos this requires GitHub Advanced Security; public repos
get secret scanning for free. Because IBM/AssetOpsBench is public, enable it
directly — it's the strongest single control and takes ~30 seconds.

---

## If a secret was already committed (important)

Rotating the secret is mandatory — scrubbing git history is not enough on its
own, because clones and forks may already have it.

1. **Revoke/rotate the leaked credential at its source immediately** (e.g.
   regenerate the API key in the provider's dashboard). Assume it's compromised.
2. Remove it from history with `git filter-repo` (preferred) or the BFG:
   ```bash
   pip install git-filter-repo
   git filter-repo --replace-text <(echo 'THE_LEAKED_VALUE==>***REMOVED***')
   git push --force --all
   ```
3. Tell collaborators to re-clone, since history was rewritten.

---

## Quick recap

- Layer 1 stops most leaks at the keyboard (opt-in, bypassable).
- Layer 2 enforces scanning in CI and blocks merges (can't be skipped).
- Layer 3 is GitHub blocking the push itself (strongest, zero-maintenance).

Enable all three for full coverage.
