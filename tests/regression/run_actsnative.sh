#!/usr/bin/env bash
# =============================================================================
# Definitive same-seed parquet validation driver.
#
# Runs ONE digi+reco execution in the Arrow-enabled ACTS image that emits BOTH:
#   - measurements.root / particles.root / tracksummary_ambi.root  (ROOT, for convert_all.py)
#   - native Arrow parquet (particles, tracker_hits, tracks, calo_hits)
# from the SAME seed-42 digitization, then runs the custom convert_all.py on
# the ROOT outputs. Because both derive from the identical digitization, the
# tracker-hit reco positions are bit-identical → an exact native-vs-v1 diff.
#
# Usage:
#   tests/regression/run_actsnative.sh <edm4hep.root> <work_dir>
# Example:
#   tests/regression/run_actsnative.sh \
#     /tmp/sim-verify/colliderml_output/higgs_portal_pu10_10evt/runs/0/edm4hep.root \
#     /tmp/sim-verify/sameseed
#
# Writes (under <work_dir>, which must be inside /tmp/sim-verify so the mount works):
#   <work_dir>/runs/0/{measurements,particles,tracksummary_ambi}.root
#   <work_dir>/runs/0/{particles,tracker_hits,tracks,calo_hits}/*.parquet   (native Arrow)
#   <work_dir>/docker_test/higgs_portal/v1/parquet/{truth,reco}/...          (v1 convert_all)
#
# Then run the comparison:
#   COLLIDERML_V1_PARQUET_DIR=<work_dir>/docker_test/higgs_portal/v1/parquet \
#   COLLIDERML_ACTSNATIVE_PARQUET_DIR=<work_dir>/runs/0 \
#     conda run -n colliderml python3 -m pytest tests/regression/ -v
# =============================================================================
set -euo pipefail

EDM4HEP="${1:?usage: run_actsnative.sh <edm4hep.root> <work_dir>}"
WORK="${2:?usage: run_actsnative.sh <edm4hep.root> <work_dir>}"
IMAGE="${ACTS_ARROW_IMAGE:-colliderml/acts-arrow:0.2.2-arrow-dev}"
REPO="${COLLIDERML_PRODUCTION_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SEED="${SEED:-42}"
MOUNT_ROOT="${MOUNT_ROOT:-/tmp/sim-verify}"

# Paths relative to the mount root (mounted as /output in the container).
rel() { realpath --relative-to="$MOUNT_ROOT" "$1"; }
WORK_REL=$(rel "$WORK")
EDM_REL=$(rel "$EDM4HEP")

mkdir -p "$WORK/runs/0"

# Seed-pinned config: ONE run, both ROOT + Arrow outputs.
cat >"$WORK/digi_config.yaml" <<EOF
campaign: "docker_test"
dataset: "higgs_portal"
version: "v1"
stage: "digitization"
common:
  output_base_dir: "/output/${WORK_REL}"
seed: ${SEED}
events: 10
threads: 1
digi: true
reco: true
num_seeds_per_spm: 40
output_root: false
output_measurements_root: true
output_particles_root: true
output_simhits_root: false
output_parquet_arrow: true
EOF

# convert_all config pointing at this run.
sed -e "s#input_base_dir: \"/output\"#input_base_dir: \"/output/${WORK_REL}\"#" \
    -e "s#output_base_dir: \"/output\"#output_base_dir: \"/output/${WORK_REL}\"#" \
    "$REPO/configs_development/docker_test/higgs_portal/convert_all.yaml" \
    > "$WORK/convert_all.yaml"

# Explicit env (P5): the bindings are cpython-313, only importable under the
# spack python; don't rely on the image's setup.sh.
SPACK_PY=/spack/opt/spack/linux-x86_64/python-venv-1.0-5z7buqdsvaex346x6rwlkt7zmxah3dmg/bin

echo "===== [1/3] digi+reco (seed=$SEED) → ROOT + Arrow ====="
docker run --rm \
  -v "$REPO":/colliderml:ro -v "$MOUNT_ROOT":/output \
  --entrypoint bash "$IMAGE" -c "
    set -e
    source /colliderml/scripts/cli/setup_container_env.sh >/dev/null 2>&1
    export PATH=$SPACK_PY:\$PATH
    export LD_LIBRARY_PATH=/opt/acts-arrow/lib:/opt/acts-arrow/lib64:\${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/opt/acts-arrow/python:\${PYTHONPATH:-}
    cd /colliderml/scripts/simulation
    python3 digi_and_reco.py \
        --config /output/${WORK_REL}/digi_config.yaml \
        --input-file /output/${EDM_REL} \
        --output /output/${WORK_REL}/runs/0
  "

# convert_all needs edm4hep.root present in the run dir (it reads particles from
# it). Relative symlink so it resolves inside the container mount.
ln -sf "$(realpath --relative-to="$WORK/runs/0" "$EDM4HEP")" "$WORK/runs/0/edm4hep.root"

echo "===== [2/3] convert_all.py on the ROOT outputs → v1 parquet ====="
docker run --rm \
  -v "$REPO":/colliderml:ro -v "$MOUNT_ROOT":/output \
  --entrypoint bash "$IMAGE" -c "
    set -e
    source /colliderml/scripts/cli/setup_container_env.sh >/dev/null 2>&1
    export PATH=$SPACK_PY:\$PATH
    export LD_LIBRARY_PATH=/opt/acts-arrow/lib:/opt/acts-arrow/lib64:\${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/opt/acts-arrow/python:\${PYTHONPATH:-}
    cd /colliderml/scripts/postprocessing
    python3 convert_all.py --config /output/${WORK_REL}/convert_all.yaml
  "

echo "===== [3/3] inventory ====="
echo "--- native Arrow parquet (runs/0) ---"
find "$WORK/runs/0" -name "*.parquet" | sort
echo "--- v1 convert_all parquet ---"
find "$WORK/docker_test" -name "*.parquet" 2>/dev/null | sort
echo ""
echo "Compare with:"
echo "  COLLIDERML_V1_PARQUET_DIR=$WORK/docker_test/higgs_portal/v1/parquet \\"
echo "  COLLIDERML_ACTSNATIVE_PARQUET_DIR=$WORK/runs/0 \\"
echo "    conda run -n colliderml python3 -m pytest tests/regression/ -v"
