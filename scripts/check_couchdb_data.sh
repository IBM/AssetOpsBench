#!/usr/bin/env bash
# Blocks committing NEW data files under src/couchdb/ (any subfolder) unless
# their path is listed in src/couchdb/.allowed_datafiles.
# Used both as a pre-commit hook (staged files) and is safe to run manually.
set -euo pipefail

EXT='\.(csv|tsv|json|jsonl|ndjson|parquet|xls|xlsx|feather|h5|hdf5|pkl|pickle|npy|npz|db|sqlite|sqlite3|avro|orc)$'
ALLOW="src/couchdb/.allowed_datafiles"

# Newly ADDED, staged files under src/couchdb that look like data.
staged=$(git diff --cached --name-only --diff-filter=A -- src/couchdb | grep -iE "$EXT" || true)

violations=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  grep -qxF "$f" "$ALLOW" 2>/dev/null || violations="${violations}${f}"$'\n'
done <<< "$staged"

if [ -n "$violations" ]; then
  echo "BLOCKED: new data files under src/couchdb/ are not allowed unless allowlisted."
  while IFS= read -r v; do [ -n "$v" ] && echo "  - $v"; done <<< "$violations"
  echo ""
  echo "If this file is intentional, add its exact path to ${ALLOW} and re-commit."
  exit 1
fi
exit 0
