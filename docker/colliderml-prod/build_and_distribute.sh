#!/usr/bin/env bash
# =============================================================================
# Build + distribute the single-container ColliderML production image.
# =============================================================================
# The whole pipeline runs in ONE podman-hpc image. podman-hpc `migrate` is
# unreliable on NERSC login sessions (sd-bus "Permission denied", exit 0), and
# compute nodes have a per-node local store, so we distribute via a CFS tarball
# that each node loads at job start (job_submission preamble does the load).
#
# Usage:
#   docker/colliderml-prod/build_and_distribute.sh [build|save|all] [TAG]
#
# Env:
#   IMAGE_TAG   default colliderml/prod:<date>
#   JOBS        default 6  (cc1plus OOMs above ~6 on the login node LTO links)
#   TARBALL_DIR default /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/images
# =============================================================================
set -eo pipefail

ACTION="${1:-all}"
IMAGE_TAG="${2:-${IMAGE_TAG:-colliderml/prod:$(date +%Y%m%d)}}"
JOBS="${JOBS:-6}"
TARBALL_DIR="${TARBALL_DIR:-/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/images}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="$TARBALL_DIR/$(echo "$IMAGE_TAG" | tr '/:' '__').tar"

mkdir -p "$TARBALL_DIR"

if [[ "$ACTION" == "build" || "$ACTION" == "all" ]]; then
  echo "[prod-image] building $IMAGE_TAG (JOBS=$JOBS) on $(hostname)"
  podman-hpc build --jobs "$JOBS" -t "$IMAGE_TAG" "$HERE"
fi

if [[ "$ACTION" == "save" || "$ACTION" == "all" ]]; then
  echo "[prod-image] saving tarball -> $TARBALL"
  podman-hpc save -o "$TARBALL" "$IMAGE_TAG"
  ls -la "$TARBALL"
  echo "[prod-image] set common.container_tarball to: $TARBALL"
  echo "[prod-image] and common.container to: localhost/$IMAGE_TAG"
fi

echo "[prod-image] done."
