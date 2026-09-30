#!/usr/bin/env bash
# Build the runtime image from `git archive HEAD`, never from the working tree.
#
#   bash benchmarks/harbor/scripts/build-runtime-image.sh
#   bash benchmarks/harbor/scripts/build-runtime-image.sh --no-cache
#   bash benchmarks/harbor/scripts/build-runtime-image.sh -t myorg/runtime:test
#
# Arguments go to `docker buildx build` unchanged. Unless they name a tag (-t)
# the image is tagged twice: assetopsbench/runtime:dev, the tag the task
# template builds FROM by default, and assetopsbench/runtime:<commit>, the
# short hash of the HEAD it was built from. :dev moves with every build; the
# commit tag does not, so `run.sh -r assetopsbench/runtime:<commit>` pins a run
# to one build. Unless they name an output (--push, --output, --load) it is
# loaded into the local image store. publish-images.sh passes --platform, its
# own tags and --push.
#
# Why an archive: base-image/Dockerfile does `COPY . .`, so a build from the
# working tree takes whatever sits there, untracked and git-ignored files
# included. An untracked reports/results_table.csv, with a ground_truth column
# for 56 private scenarios, once reached the image that way, readable by any
# agent running code in `main`. The archive holds only what HEAD commits and no
# .git, whose blobs would hold the answers .dockerignore drops. The commit
# travels as the AOB_COMMIT build arg instead, which the Dockerfile requires.
#
# Uncommitted changes and new files are not built. The script warns about both;
# commit them first to include them.
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

if ! docker buildx version >/dev/null 2>&1; then
  printf 'docker buildx is required (Docker Desktop and docker-ce ship it)\n' >&2
  exit 1
fi

if ! git -C "$repo_root" diff --quiet HEAD --; then
  printf 'warning: tracked files differ from HEAD; building %s without those changes\n' \
    "${commit:0:7}" >&2
fi
untracked="$(git -C "$repo_root" ls-files --others --exclude-standard | wc -l | tr -d ' ')"
if (( untracked > 0 )); then
  printf 'warning: %s untracked file(s) are not in the image; `git add` and commit any it needs\n' \
    "$untracked" >&2
fi

context="$(mktemp -d "${TMPDIR:-/tmp}/aob-runtime-context.XXXXXX")"
trap 'rm -rf "$context"' EXIT
git -C "$repo_root" archive --format=tar HEAD | tar -x -C "$context"

has_tag=false
has_output=false
for arg in "$@"; do
  case "$arg" in
    -t | --tag | --tag=*) has_tag=true ;;
    --push | --load | --output | --output=* | -o) has_output=true ;;
  esac
done
defaults=()
$has_tag || defaults+=(-t assetopsbench/runtime:dev -t "assetopsbench/runtime:${commit:0:7}")
$has_output || defaults+=(--load)

printf 'Building the runtime image from %s (git archive)\n' "${commit:0:7}" >&2
docker buildx build \
  -f "$context/benchmarks/harbor/base-image/Dockerfile" \
  --build-arg "AOB_COMMIT=$commit" \
  ${defaults[@]+"${defaults[@]}"} "$@" \
  "$context"
