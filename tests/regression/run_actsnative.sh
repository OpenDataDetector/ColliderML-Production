#!/usr/bin/env bash
# =============================================================================
# Re-run the digi+reco stage with the Arrow-enabled ACTS image, producing
# parquet directly via the native Arrow writers. Inputs are the same edm4hep.root
# the legacy pipeline already produced, so this is a pure A/B test of the
# postprocessing path — Pythia + DDSim runs are reused.
#
# Usage:
#   tests/regression/run_actsnative.sh /tmp/sim-verify/colliderml_output/higgs_portal_pu10_10evt
#
# Reads:
#   <run_dir>/runs/0/edm4hep.root
# Writes:
#   <run_dir>/actsnative_parquet/{particles,tracker_hits,tracks,calo_hits}/*.parquet
# =============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir>" >&2
    exit 1
fi

RUN_DIR="$1"
IMAGE="${ACTS_ARROW_IMAGE:-colliderml/acts-arrow:0.2.2-arrow-dev}"
CONFIG_DIR="${COLLIDERML_PRODUCTION_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
DIGI_CONFIG="${DIGI_CONFIG:-configs_development/docker_test/higgs_portal/digitization_config.yaml}"

OUTPUT_DIR="$RUN_DIR/actsnative_parquet"
mkdir -p "$OUTPUT_DIR"

# Compose an override config that flips on the parquet writer. We keep
# everything else the same as the standard digitization config so the
# reconstruction path is byte-identical between the two runs.
OVERLAY="$OUTPUT_DIR/overlay_config.yaml"
cat >"$OVERLAY" <<EOF
output_parquet_arrow: true
output_root: false
output_measurements_root: false
output_particles_root: false
output_simhits_root: false
EOF

echo "===== Re-running digi_and_reco with Arrow writers ====="
echo "  image:   $IMAGE"
echo "  run dir: $RUN_DIR"
echo "  output:  $OUTPUT_DIR"
echo "  digi:    $DIGI_CONFIG"
echo "  overlay: $OVERLAY"

docker run --rm \
    -v "$CONFIG_DIR":/colliderml:ro \
    -v "$RUN_DIR":/output \
    -v "$OVERLAY":/overlay.yaml:ro \
    --entrypoint bash \
    "$IMAGE" \
    -lc '
        set -euo pipefail
        # Use the Arrow-enabled ACTS in /opt/acts-arrow, fall back to spack
        if [ -f /opt/acts-arrow/setup.sh ]; then
            source /opt/acts-arrow/setup.sh
        fi
        # Reuse the rest of the container environment.
        source /colliderml/scripts/cli/setup_container_env.sh >/dev/null 2>&1
        cd /colliderml
        python3 scripts/simulation/digi_and_reco.py \
            --config '"$DIGI_CONFIG"' \
            --config-overlay /overlay.yaml \
            --input-file /output/runs/0/edm4hep.root \
            --output /output/actsnative_parquet
    '

echo "===== Done. ACTS-native parquet at: $OUTPUT_DIR ====="
ls -lh "$OUTPUT_DIR"/*/ 2>/dev/null | head -20
