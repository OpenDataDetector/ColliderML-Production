#!/usr/bin/env bash
# =============================================================================
# ColliderML Docker Runner
# =============================================================================
# Run a single ColliderML pipeline stage inside the ODD software container.
#
# Usage:
#   ./scripts/cli/run_docker.sh <stage_script> <config.yaml> [extra args...]
#
# Examples:
#   # Pythia generation
#   ./scripts/cli/run_docker.sh simulation/pythia_gen.py \
#       configs_development/docker_test/higgs_portal/pythia_config.yaml
#
#   # Detector simulation
#   ./scripts/cli/run_docker.sh simulation/ddsim_run.py \
#       configs_development/docker_test/higgs_portal/simulation_config.yaml
#
#   # Override number of events
#   ./scripts/cli/run_docker.sh simulation/pythia_gen.py \
#       configs_development/docker_test/higgs_portal/pythia_config.yaml --events 100
#
# Environment Variables:
#   CONTAINER_IMAGE  - Docker image (default: ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0)
#   OUTPUT_DIR       - Host directory for output (default: ./output)
#   SEED             - Random seed (default: 42)
#   RUN_ID           - Run subdirectory (default: 0)
# =============================================================================

set -euo pipefail

# --- Configuration ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-0}"

# --- Validate arguments ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <stage_script> <config.yaml> [extra args...]"
    echo ""
    echo "Stage scripts (relative to scripts/):"
    echo "  simulation/pythia_gen.py          - Pythia8 event generation"
    echo "  simulation/madgraph_init.py       - MadGraph process compilation"
    echo "  simulation/madgraph_gen.py        - MadGraph event generation"
    echo "  simulation/ddsim_run.py           - Geant4 detector simulation"
    echo "  simulation/digi_and_reco.py       - Digitization + reconstruction"
    echo "  postprocessing/convert_all.py     - Convert to parquet"
    exit 1
fi

STAGE_SCRIPT="$1"
CONFIG_PATH="$2"
shift 2  # Remaining args passed through to the stage script

# Get the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Validate files exist
if [ ! -f "$REPO_ROOT/$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $REPO_ROOT/$CONFIG_PATH"
    exit 1
fi
if [ ! -f "$REPO_ROOT/scripts/$STAGE_SCRIPT" ]; then
    echo "ERROR: Stage script not found: $REPO_ROOT/scripts/$STAGE_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "ColliderML Docker Runner"
echo "=============================================="
echo "Container:  $CONTAINER_IMAGE"
echo "Stage:      $STAGE_SCRIPT"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Seed:       $SEED"
echo "Run ID:     $RUN_ID"
echo "Extra args: $*"
echo "=============================================="

# --- Ensure cache directory exists ---
CACHE_DIR="$REPO_ROOT/.cache"
mkdir -p "$CACHE_DIR"

# Clone ODD if not present (done on host where network access works)
if [ ! -f "$CACHE_DIR/odd/xml/OpenDataDetector.xml" ]; then
    echo "Cloning OpenDataDetector geometry..."
    git clone --depth 1 https://github.com/acts-project/OpenDataDetector.git "$CACHE_DIR/odd" 2>/dev/null \
        || echo "WARNING: Failed to clone ODD. Simulation stages will fail."
fi

# --- Run ---
# Mounts .cache from host for persistent storage of:
#   - ODD detector geometry (cloned from GitHub)
#   - Geant4 physics datasets (~2 GB, downloaded inside container)
#   - pip-installed Python packages
docker run --rm \
    -v "$REPO_ROOT":/workspace \
    -v "$OUTPUT_DIR":/output \
    -v "$CACHE_DIR":/cache \
    -e COLLIDERML_CACHE=/cache \
    -e HTTP_PROXY="${HTTP_PROXY:-}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-}" \
    -e http_proxy="${http_proxy:-}" \
    -e https_proxy="${https_proxy:-}" \
    -e NO_PROXY="${NO_PROXY:-}" \
    "$CONTAINER_IMAGE" \
    -c "
        source /workspace/scripts/cli/setup_container_env.sh

        cd /workspace/scripts/$(dirname "$STAGE_SCRIPT")
        python3 $(basename "$STAGE_SCRIPT") \
            --config /workspace/$CONFIG_PATH \
            --output /output/runs \
            --output-subdir $RUN_ID \
            --seed $SEED \
            $*
    "

echo ""
echo "=============================================="
echo "Run complete! Output files:"
echo "=============================================="
find "$OUTPUT_DIR/runs/$RUN_ID/" -type f -exec ls -lh {} \; 2>/dev/null || echo "(no output files found)"
