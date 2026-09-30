#!/usr/bin/env bash
# Build the runtime image from a clean one-commit clone of HEAD, never from the
# working tree.
#
#   bash benchmarks/harbor/scripts/build-runtime-image.sh
#   bash benchmarks/harbor/scripts/build-runtime-image.sh -t assetopsbench/runtime:abc1234 --load
#
# With no arguments it builds and loads assetopsbench/runtime:dev, the tag the
# task template builds FROM. Arguments replace that default and go to
# `docker buildx build` unchanged; publish-images.sh passes --platform and --push.
#
# Why a clone: base-image/Dockerfile does `COPY . .`, so a build from the working
# tree takes whatever sits there, untracked and git-ignored files included. An
# untracked reports/results_table.csv, with a ground_truth column for 56 private
# scenarios, once reached the image that way, readable by any agent running code
# in `main`. The clone holds only what HEAD commits, plus a one-commit .git so
# `git rev-parse HEAD` in the container still names the commit each trial ran
# against (StirrupAgent.get_version_command). .dockerignore still applies inside
# the clone as a second line of defence.
#
# Uncommitted changes are not built. The script warns when tracked files differ
# from HEAD; commit them first to include them.
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

if ! git -C "$repo_root" diff --quiet HEAD --; then
  printf 'warning: tracked files differ from HEAD; building %s without those changes\n' \
    "${commit:0:7}" >&2
fi

context="$(mktemp -d "${TMPDIR:-/tmp}/aob-runtime-context.XXXXXX")"
trap 'rm -rf "$context"' EXIT

# file:// rather than a plain path: a local path makes git hardlink the whole
# object store and ignore --depth. Fetching HEAD also covers a detached HEAD.
git -C "$context" init -q
git -C "$context" fetch -q --depth 1 "file://$repo_root" HEAD
git -C "$context" checkout -q --detach FETCH_HEAD
# FETCH_HEAD records the host path the clone came from; the image needs none of it.
rm -f "$context/.git/FETCH_HEAD"

cloned="$(git -C "$context" rev-parse HEAD)"
if [[ "$cloned" != "$commit" ]]; then
  printf 'clean clone is at %s, expected %s\n' "$cloned" "$commit" >&2
  exit 1
fi

if (( $# == 0 )); then
  set -- -t assetopsbench/runtime:dev --load
fi

printf 'Building the runtime image from %s (clean clone)\n' "${commit:0:7}" >&2
docker buildx build -f "$context/benchmarks/harbor/base-image/Dockerfile" "$@" "$context"
