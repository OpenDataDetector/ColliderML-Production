#!/usr/bin/env bash
# =============================================================================
# Run one ColliderML pipeline stage in a container (docker, podman or podman-hpc)
# =============================================================================
# Usage:
#   ./scripts/cli/run_docker.sh <stage_script> <config.yaml> [stage script args...]
#
# <stage_script> is relative to scripts/, <config.yaml> relative to the repository.
#
# Examples:
#   ./scripts/cli/run_docker.sh simulation/pythia_gen.py \
#       configs_development/docker_test/higgs_portal/pythia_config.yaml --events 5
#   ./scripts/cli/run_docker.sh simulation/pandora_reco.py \
#       configs_development/docker_test/ttbar/pandora_reco_config.yaml
#
# The stage runs in the sim image, except pandora_reco.py and convert_reco_tables.py,
# which run in the reco image (see scripts/cli/docker_images.env).
#
# Environment variables:
#   OUTPUT_DIR         host output directory (default: ./output), mounted as /output
#   CACHE_DIR          host cache directory (default: <repo>/.cache), mounted as /cache;
#                      holds the Geant4 datasets (about 2 GB), downloaded on the first run
#   SEED               random seed (default: 42)
#   RUN_ID             run subdirectory under <output>/runs (default: 0)
#   SIM_IMAGE          sim image (CONTAINER_IMAGE is accepted as an older name)
#   RECO_IMAGE         reco image
#   CONTAINER_RUNTIME  docker, podman-hpc or podman (default: the first one found)
#   STAGE_NAME         label in <output>/pipeline_timing.csv (default: the script name)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -lt 2 ]; then
    sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi

STAGE_SCRIPT="$1"
CONFIG_PATH="$2"
shift 2

SIM_IMAGE="${SIM_IMAGE:-${CONTAINER_IMAGE:-}}"
[ -n "$SIM_IMAGE" ] || unset SIM_IMAGE
# shellcheck source=docker_images.env
source "$SCRIPT_DIR/docker_images.env"

OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-0}"
STAGE_NAME="${STAGE_NAME:-$(basename "$STAGE_SCRIPT" .py)}"

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: RUN_ID must be a plain name (letters, digits, '_', '.', '-'), got '$RUN_ID'" >&2
    exit 1
fi
if [ ! -f "$REPO_ROOT/$CONFIG_PATH" ]; then
    echo "ERROR: config file not found: $REPO_ROOT/$CONFIG_PATH" >&2
    exit 1
fi
if [ ! -f "$REPO_ROOT/scripts/$STAGE_SCRIPT" ]; then
    echo "ERROR: stage script not found: $REPO_ROOT/scripts/$STAGE_SCRIPT" >&2
    exit 1
fi

if [ -z "${CONTAINER_RUNTIME:-}" ]; then
    # podman-hpc before podman: where the wrapper exists, plain podman is on PATH too
    # but does not see the migrated images
    for candidate in docker podman-hpc podman; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CONTAINER_RUNTIME="$candidate"
            break
        fi
    done
fi
if [ -z "${CONTAINER_RUNTIME:-}" ]; then
    echo "ERROR: no container runtime found (looked for docker, podman-hpc, podman)" >&2
    exit 1
fi

case "$(basename "$STAGE_SCRIPT")" in
    pandora_reco.py|convert_reco_tables.py) ENV_KIND=reco; IMAGE="$RECO_IMAGE" ;;
    *)                                      ENV_KIND=sim;  IMAGE="$SIM_IMAGE" ;;
esac

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"

echo "--- $STAGE_NAME: $STAGE_SCRIPT ($ENV_KIND image $IMAGE, $CONTAINER_RUNTIME)"

# Proxy variables set on the host are passed through for the first-run downloads
# (Geant4 datasets).
PROXY_ARGS=()
for name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [ -n "${!name:-}" ]; then
        PROXY_ARGS+=(-e "$name=${!name}")
    fi
done

# The stage's settings reach the container as positional parameters of a fixed
# command, so no value is re-read as shell source.
# shellcheck disable=SC2016
IN_CONTAINER='
    env_kind=$1 stage_name=$2 stage_script=$3 config_path=$4 run_id=$5 seed=$6
    shift 6
    source /workspace/scripts/cli/docker_stage_env.sh "$env_kind" || exit 1
    cd "/workspace/scripts/$(dirname "$stage_script")" || exit 1
    exec python3 /workspace/scripts/cli/stage_timer.py \
        --record /output/pipeline_timing.csv --stage "$stage_name" -- \
        python3 "$(basename "$stage_script")" \
            --config "/workspace/$config_path" \
            --output /output/runs \
            --output-subdir "$run_id" \
            --seed "$seed" \
            "$@"
'

"$CONTAINER_RUNTIME" run --rm \
    -v "$REPO_ROOT":/workspace \
    -v "$OUTPUT_DIR":/output \
    -v "$CACHE_DIR":/cache \
    -e COLLIDERML_CACHE=/cache \
    "${PROXY_ARGS[@]}" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "$IN_CONTAINER" stage \
    "$ENV_KIND" "$STAGE_NAME" "$STAGE_SCRIPT" "$CONFIG_PATH" "$RUN_ID" "$SEED" "$@"
