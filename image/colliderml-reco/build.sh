#!/bin/bash
# Build the ColliderML reconstruction image.
#
#   ./build.sh [VERSION]        VERSION defaults to today's date (YYYYMMDD)
#
# Environment:
#   CONTAINER_ENGINE   podman (default), docker or podman-hpc
#   IMAGE_NAME         local image name (default: colliderml/reco)
#   BUILD_JOBS         parallel compile jobs (default: all cores)
#
# Extra arguments after VERSION go to "<engine> build", for example
#   ./build.sh 20261127-reference --build-arg PANDORA_VARIANT=reference
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -gt 0 && "$1" != -* ]]; then
  version="$1"
  shift
else
  version="$(date +%Y%m%d)"
fi
engine="${CONTAINER_ENGINE:-podman}"
image="${IMAGE_NAME:-colliderml/reco}:${version}"

(cd patches && sha256sum --check --quiet --strict SHA256SUMS)

"$engine" build \
  --file Containerfile \
  --tag "$image" \
  --build-arg RECO_VERSION="$version" \
  --build-arg BUILD_JOBS="${BUILD_JOBS:-}" \
  "$@" \
  .

echo "built $image"
"$engine" image inspect --format 'image id: {{.Id}}' "$image"
