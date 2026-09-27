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
# The runtime build context is the whole repository, so .dockerignore decides
# what lands in the published image. It keeps .env out and keeps .git in, the
# latter because StirrupAgent.get_version_command runs `git rev-parse` inside
# the container to record the commit each trial ran against. Check it before
# publishing if you have added anything sensitive to the tree.
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

if [ -f .env ]; then
    echo "note: .env exists and is excluded by .dockerignore, so it stays out" >&2
fi

echo "==> ${NAMESPACE}/runtime:${TAG} for ${PLATFORMS}"
docker buildx build \
    --platform "${PLATFORMS}" \
    -t "${NAMESPACE}/runtime:${TAG}" \
    -t "${NAMESPACE}/runtime:latest" \
    -f benchmarks/harbor/base-image/Dockerfile \
    --push .

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
  docker tag  ${NAMESPACE}/runtime:latest assetopsbench/runtime:dev

The second line matters. benchmarks/harbor/template/environment/Dockerfile
references the local tag assetopsbench/runtime:dev through its
AOB_RUNTIME_IMAGE build arg, so the pulled image has to carry that tag. If you
publish under a different namespace, change that default instead of asking
every user to retag.

For the code track, point the overlay at the published image rather than a tar:

  export AOB_CODE_IMAGE=${NAMESPACE}/code:latest

NOTE
