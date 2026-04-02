#!/usr/bin/env bash
# =============================================================================
# ColliderML Full Pipeline Runner (Docker)
# =============================================================================
# Runs the complete ColliderML pipeline for a given channel inside Docker.
#
# Usage:
#   ./scripts/cli/run_pipeline_docker.sh --channel higgs_portal
#   ./scripts/cli/run_pipeline_docker.sh --channel ttbar
#
# This orchestrates all stages in sequence:
#   Higgs portal:  pythia → simulation → digitization → convert_all
#   ttbar:         madgraph_init → madgraph_gen → pythia → sim → digi → convert
#
# Environment Variables:
#   CONTAINER_IMAGE  - Docker image (default: ghcr.io/opendatadetector/sw:0.2.2_...)
#   OUTPUT_DIR       - Host output directory (default: ./output)
#   SEED             - Random seed (default: 42)
#   EVENTS           - Override event count (optional)
# =============================================================================

set -euo pipefail

# --- Parse arguments ---
CHANNEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel) CHANNEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CHANNEL" ]; then
    echo "Usage: $0 --channel <higgs_portal|ttbar>"
    exit 1
fi

# --- Configuration ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
SEED="${SEED:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_BASE="configs_development/docker_test/$CHANNEL"

# Validate channel
if [ ! -d "$REPO_ROOT/$CONFIG_BASE" ]; then
    echo "ERROR: Config directory not found: $CONFIG_BASE"
    echo "Available channels:"
    ls "$REPO_ROOT/configs_development/docker_test/" 2>/dev/null
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- Ensure cache directory exists ---
CACHE_DIR="$REPO_ROOT/.cache"
mkdir -p "$CACHE_DIR"

# Clone ODD v4.0.4 if not present (done on host where network access works)
if [ ! -f "$CACHE_DIR/odd-v4/xml/OpenDataDetector.xml" ]; then
    echo "Cloning OpenDataDetector v4.0.4 from CERN GitLab..."
    git clone --depth 1 --branch v4.0.4 \
        https://gitlab.cern.ch/acts/OpenDataDetector.git "$CACHE_DIR/odd-v4" 2>/dev/null \
        || echo "WARNING: Failed to clone ODD. Simulation stages will fail."
fi

# Clone MG5aMC_PY8_interface if not present (needed for MadGraph+Pythia8 shower)
if [ ! -f "$CACHE_DIR/MG5aMC_PY8_interface/MG5aMC_PY8_interface.cc" ]; then
    echo "Cloning MG5aMC_PY8_interface from GitHub..."
    git clone --depth 1 \
        https://github.com/mg5amcnlo/MG5aMC_PY8_interface.git "$CACHE_DIR/MG5aMC_PY8_interface" 2>/dev/null \
        || echo "WARNING: Failed to clone MG5aMC_PY8_interface. MadGraph shower will not work."
fi

# Download bc .deb if not cached (no apt-get inside container without network)
if [ ! -f "$CACHE_DIR"/bc_*.deb ]; then
    echo "Downloading bc package..."
    (cd "$CACHE_DIR" && apt-get download bc 2>/dev/null) \
        || echo "WARNING: Failed to download bc. MadGraph shower will not work."
fi

echo "============================================================"
echo "ColliderML Full Pipeline: $CHANNEL"
echo "============================================================"
echo "Container: $CONTAINER_IMAGE"
echo "Output:    $OUTPUT_DIR"
echo "Configs:   $CONFIG_BASE/"
echo "============================================================"

# --- Helper function to run a stage ---
run_stage() {
    local stage_name="$1"
    local stage_script="$2"
    local config_file="$3"
    shift 3
    local extra_args=("$@")

    echo ""
    echo "============================================================"
    echo "  STAGE: $stage_name"
    echo "============================================================"

    if [ ! -f "$REPO_ROOT/$config_file" ]; then
        echo "  SKIP: Config not found: $config_file"
        return 0
    fi

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

            cd /workspace/scripts/$(dirname "$stage_script")
            python3 $(basename "$stage_script") \
                --config /workspace/$config_file \
                --output /output/runs \
                --output-subdir 0 \
                --seed $SEED \
                ${extra_args[*]:-}
        "

    echo "  DONE: $stage_name"
}

# --- Run pipeline ---
case "$CHANNEL" in
    higgs_portal)
        run_stage "Pythia Generation" \
            "simulation/pythia_gen.py" \
            "$CONFIG_BASE/pythia_config.yaml"

        run_stage "Detector Simulation" \
            "simulation/ddsim_run.py" \
            "$CONFIG_BASE/simulation_config.yaml"

        run_stage "Digitization & Reconstruction" \
            "simulation/digi_and_reco.py" \
            "$CONFIG_BASE/digitization_config.yaml"

        run_stage "Parquet Conversion" \
            "postprocessing/convert_all.py" \
            "$CONFIG_BASE/convert_all.yaml"
        ;;

    ttbar)
        run_stage "MadGraph Init" \
            "simulation/madgraph_init.py" \
            "$CONFIG_BASE/madgraph_init_config.yaml"

        run_stage "MadGraph Generation" \
            "simulation/madgraph_gen.py" \
            "$CONFIG_BASE/madgraph_generation_config.yaml"

        run_stage "Pythia Generation" \
            "simulation/pythia_gen.py" \
            "$CONFIG_BASE/pythia_config.yaml"

        run_stage "Detector Simulation" \
            "simulation/ddsim_run.py" \
            "$CONFIG_BASE/simulation_config.yaml"

        run_stage "Digitization & Reconstruction" \
            "simulation/digi_and_reco.py" \
            "$CONFIG_BASE/digitization_config.yaml"

        run_stage "Parquet Conversion" \
            "postprocessing/convert_all.py" \
            "$CONFIG_BASE/convert_all.yaml"
        ;;

    *)
        echo "ERROR: Unknown channel: $CHANNEL"
        echo "Available: higgs_portal, ttbar"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Pipeline complete for: $CHANNEL"
echo "============================================================"
echo "Output directory:"
find "$OUTPUT_DIR/runs/0/" -type f -exec ls -lh {} \; 2>/dev/null || echo "(no output)"
