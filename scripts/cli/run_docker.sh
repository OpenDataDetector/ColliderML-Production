#!/usr/bin/env bash
# =============================================================================
# ColliderML Docker Runner
# =============================================================================
# Run ColliderML pipeline stages inside the ODD software container.
#
# Usage:
#   ./scripts/cli/run_docker.sh <config.yaml> [options]
#
# Examples:
#   # Run Higgs portal Pythia generation (10 events, test config)
#   ./scripts/cli/run_docker.sh configs_development/docker_test/higgs_portal/pythia_config.yaml
#
#   # Override number of events
#   ./scripts/cli/run_docker.sh configs_development/docker_test/higgs_portal/pythia_config.yaml --events 100
#
#   # Custom output directory
#   OUTPUT_DIR=/my/output ./scripts/cli/run_docker.sh configs_development/docker_test/higgs_portal/pythia_config.yaml
#
# Environment Variables:
#   CONTAINER_IMAGE  - Docker image to use (default: ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0)
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
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [additional pythia_gen.py args...]"
    echo ""
    echo "Example:"
    echo "  $0 configs_development/docker_test/higgs_portal/pythia_config.yaml"
    exit 1
fi

CONFIG_PATH="$1"
shift  # Remaining args passed through to pythia_gen.py

# Get the repo root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Validate config exists
if [ ! -f "$REPO_ROOT/$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $REPO_ROOT/$CONFIG_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "ColliderML Docker Runner"
echo "=============================================="
echo "Container:  $CONTAINER_IMAGE"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Seed:       $SEED"
echo "Run ID:     $RUN_ID"
echo "Extra args: $*"
echo "=============================================="

# --- Run ---
docker run --rm \
    -v "$REPO_ROOT":/workspace:ro \
    -v "$OUTPUT_DIR":/output \
    "$CONTAINER_IMAGE" \
    -c "
        # Set up the container environment
        source /workspace/scripts/cli/setup_container_env.sh

        # Run the Pythia generation stage
        cd /workspace/scripts/simulation
        python3 pythia_gen.py \
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
ls -lh "$OUTPUT_DIR/runs/$RUN_ID/" 2>/dev/null || echo "(no output files found)"
