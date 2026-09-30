#!/usr/bin/env bash
# Build and publish the images the Harbor tasks depend on, for both
# architectures.
#
#   ./benchmarks/harbor/scripts/publish-images.sh quay.io/assetopsbench v0.1.0
#
# Run from the repository root. Requires `docker login` to that registry and a
# buildx builder that can do multi-platform builds:
#
#   docker buildx create --name aob --use --bootstrap
#
# Two images:
#   <namespace>/runtime  the repo, its uv environment and the shared scenario
#                        data. Every task image layers its scenario onto this.
#   <namespace>/code     the sandbox for the code track (numpy, pandas, scipy).
#                        Only needed by the code-sandbox overlay.
#
# The runtime image is built by build-runtime-image.sh from `git archive HEAD`,
# so uncommitted changes are not published.
set -euo pipefail

NAMESPACE="${1:?usage: publish-images.sh <namespace, e.g. quay.io/assetopsbench> [tag]}"
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

# Unlike TAG and latest, the commit tag names the exact build a run used.
COMMIT="$(git rev-parse HEAD | cut -c1-7)"

echo "==> ${NAMESPACE}/runtime:${TAG} (${COMMIT}) for ${PLATFORMS}"
bash benchmarks/harbor/scripts/build-runtime-image.sh \
    --platform "${PLATFORMS}" \
    -t "${NAMESPACE}/runtime:${TAG}" \
    -t "${NAMESPACE}/runtime:${COMMIT}" \
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

Published. Users now pull instead of building. To run on exactly this build:

  docker pull ${NAMESPACE}/runtime:${COMMIT}
  export AOB_RUNTIME_IMAGE=${NAMESPACE}/runtime:${COMMIT}

or use ${NAMESPACE}/runtime:latest to follow the newest publish.
benchmarks/harbor/run.sh takes the image as -r.

For the code track, point the overlay at the published image rather than a tar:

  export AOB_CODE_IMAGE=${NAMESPACE}/code:latest

NOTE
