#!/usr/bin/env bash
# Build and publish the images the Harbor tasks depend on, for both
# architectures, so nobody has to build them locally.
#
#   ./benchmarks/harbor/scripts/publish-images.sh assetopsbench v0.1.0
#
# Run from the repository root. Requires `docker login` and a buildx builder
# that can do multi-platform builds:
#
#   docker buildx create --name aob --use --bootstrap
#
# Two images:
#   <namespace>/runtime  the repo, its uv environment and the shared corpus.
#                        Every task image layers its scenario onto this.
#   <namespace>/code     the sandbox for the code track (numpy, pandas, scipy).
#                        Only needed by the code-sandbox overlay.
#
# The runtime image is built by build-runtime-image.sh from `git archive HEAD`,
# so only committed files can land in the published image, and .dockerignore
# then drops env files and scenario answers from those. The image carries no
# .git; the commit it was built from is /opt/aob/.aob-commit, which
# StirrupAgent.get_version_command records for each trial. Uncommitted changes
# are not published; commit them first.
set -euo pipefail

NAMESPACE="${1:?usage: publish-images.sh <dockerhub-namespace> [tag]}"
TAG="${2:-dev}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

if [ ! -f benchmarks/harbor/base-image/Dockerfile ]; then
    echo "run this from the repository root" >&2
    exit 1
fi

if [ ! -f src/agent/stirrup_agent/Dockerfile.code ]; then
    echo "missing src/agent/stirrup_agent/Dockerfile.code" >&2
    exit 1
fi

if ! docker buildx inspect >/dev/null 2>&1; then
    echo "no buildx builder; run: docker buildx create --name aob --use --bootstrap" >&2
    exit 1
fi

echo "==> ${NAMESPACE}/runtime:${TAG} for ${PLATFORMS}"
bash benchmarks/harbor/scripts/build-runtime-image.sh \
    --platform "${PLATFORMS}" \
    -t "${NAMESPACE}/runtime:${TAG}" \
    -t "${NAMESPACE}/runtime:latest" \
    --push

echo "==> ${NAMESPACE}/code:${TAG} for ${PLATFORMS}"
docker buildx build \
    --platform "${PLATFORMS}" \
    -t "${NAMESPACE}/code:${TAG}" \
    -t "${NAMESPACE}/code:latest" \
    -f src/agent/stirrup_agent/Dockerfile.code \
    --push src/agent/stirrup_agent

cat <<NOTE

Published. Users now pull instead of building:

  docker pull ${NAMESPACE}/runtime:latest
  export AOB_RUNTIME_IMAGE=${NAMESPACE}/runtime:latest

Each task's docker-compose.yaml passes AOB_RUNTIME_IMAGE to its Dockerfile as
a build arg, so harbor run builds FROM that image; unset, it falls back to the
local tag assetopsbench/runtime:dev. benchmarks/harbor/run.sh takes it as -r.

For the code track, point the overlay at the published image rather than a tar:

  export AOB_CODE_IMAGE=${NAMESPACE}/code:latest

NOTE
