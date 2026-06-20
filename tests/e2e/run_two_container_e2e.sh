#!/usr/bin/env bash
# =============================================================================
# Charged-PF TWO-CONTAINER end-to-end test (sim image -> reco image).
# =============================================================================
# Runs the REAL production stage scripts directly in the two published images,
# with file handoff, to prove the charged particle-flow chain end-to-end:
#
#   SIM image  (acts-arrow): ddsim_run.py --single-particle  -> edm4hep.root
#                            digi_and_reco.py (output_sim_with_tracks) -> sim_with_tracks.root
#                                (ACTS digi + seeding + CKF + ambiguity, then the new
#                                 EDM4hepTrackOutputConverter -> ActsTracks edm4hep collection)
#   RECO image (pandora):    pandora_reco.py (track_collection=ActsTracks) -> reco_edm4hep.root
#
# Asserts: ActsTracks present in sim_with_tracks.root, and GaudiPandoraPFOs /
# GaudiPandoraClusters present in reco_edm4hep.root (charged PF built FROM tracks,
# not calo-only). This is the thing the per-image smoke tests can't cover.
#
# Geometry: sim and reco must share the calibrated azaborow ODD
# (addLayeredCalo_MuonCoil). The reco image bakes it at /opt/odd-install; here we build a
# matching one in the sim image (same XML/branch -> identical cellIDs) and point
# ddsim_run.py (ODD_COMPACT_FILE) / digi_and_reco.py (ODD_GEO_DIR) at it.
#
# Usage:  tests/e2e/run_two_container_e2e.sh <sim_image> <reco_image> [n_events]
# Needs:  docker; both images already pulled; a runner that can hold both (~25 GB each)
#         and run Geant4. NOT for per-PR CI — dispatch/nightly only.
# =============================================================================
set -euo pipefail

SIM_IMAGE="${1:?usage: run_two_container_e2e.sh <sim_image> <reco_image> [n_events]}"
RECO_IMAGE="${2:?usage: run_two_container_e2e.sh <sim_image> <reco_image> [n_events]}"
NEV="${3:-2}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d)"; chmod 777 "$WORK"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/runs/0"
echo "sim=$SIM_IMAGE  reco=$RECO_IMAGE  nev=$NEV  repo=$REPO  work=$WORK"

ODD_REPO="https://gitlab.cern.ch/azaborow/OpenDataDetector.git"
ODD_REF="addLayeredCalo_MuonCoil"

# Spack python (the acts-arrow bindings are cpython-specific; don't rely on image setup.sh).
SPACK_PY=/spack/opt/spack/linux-x86_64/python-venv-1.0-5z7buqdsvaex346x6rwlkt7zmxah3dmg/bin

# Stage configs (gun pi-, ~NEV events). Written to the shared work dir.
cat > "$WORK/ddsim.yaml" <<EOF
campaign: "e2e"
dataset: "single_pim"
version: "v1"
stage: "simulation"
events: ${NEV}
threads: 1
seed: 42
single_particle: true
gun_particle: "pi-"
gun_energy: 10.0
gun_distribution: "uniform"
gun_theta_min: "60*deg"
gun_theta_max: "120*deg"
common:
  output_base_dir: "/output"
EOF
cat > "$WORK/digi.yaml" <<EOF
campaign: "e2e"
dataset: "single_pim"
version: "v1"
stage: "digitization"
events: ${NEV}
threads: 1
seed: 42
digi: true
reco: true
output_sim_with_tracks: true     # <-- emit the edm4hep ActsTracks file for Pandora
num_seeds_per_spm: 40
common:
  output_base_dir: "/output"
EOF
cat > "$WORK/pandora.yaml" <<EOF
campaign: "e2e"
dataset: "single_pim"
version: "v1"
stage: "pandora_reco"
events: -1
track_collection: "ActsTracks"   # charged PF (the default + required mode)
common:
  output_base_dir: "/output"
EOF

# ---------------------------------------------------------------------------
# Phase 1 (SIM image): build azaborow ODD, ddsim gun -> edm4hep.root,
#                      digi_and_reco -> sim_with_tracks.root (with ActsTracks).
# ---------------------------------------------------------------------------
echo "::group::sim image — azaborow ODD + ddsim + digi_and_reco (ACTS tracking -> ActsTracks)"
docker run --rm -v "$REPO":/colliderml:ro -v "$WORK":/output \
  -e NEV="$NEV" -e ODD_REPO="$ODD_REPO" -e ODD_REF="$ODD_REF" -e SPACK_PY="$SPACK_PY" \
  --entrypoint bash "$SIM_IMAGE" -c '
    set -euo pipefail
    source /colliderml/scripts/cli/setup_container_env.sh >/dev/null 2>&1 || true
    source /opt/acts-arrow/setup.sh >/dev/null 2>&1 || true
    export PATH="$SPACK_PY:$PATH"
    export LD_LIBRARY_PATH=/opt/acts-arrow/lib:/opt/acts-arrow/lib64:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/opt/acts-arrow/python:${PYTHONPATH:-}
    # Build the calibrated azaborow ODD against this image (same XML/branch as the reco
    # image bakes -> identical cellIDs). Provides xml/ + data/ + config/ for ODD_GEO_DIR.
    git clone --depth 1 --single-branch --branch "$ODD_REF" "$ODD_REPO" /tmp/odd-src
    cmake -S /tmp/odd-src -B /tmp/odd-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/odd-install
    cmake --build /tmp/odd-build -j"$(nproc)" --target install
    export LD_LIBRARY_PATH=/tmp/odd-install/lib:/tmp/odd-install/lib64:$LD_LIBRARY_PATH
    export ODD_GEO_DIR=/tmp/odd-src
    export ODD_COMPACT_FILE=/tmp/odd-src/xml/OpenDataDetector.xml
    test -f "$ODD_COMPACT_FILE"
    cd /colliderml/scripts/simulation
    python3 ddsim_run.py --config /output/ddsim.yaml --single-particle --output /output/runs/0
    test -s /output/runs/0/edm4hep.root
    python3 digi_and_reco.py --config /output/digi.yaml \
        --input-file /output/runs/0/edm4hep.root --output /output/runs/0
    ls -l /output/runs/0/sim_with_tracks.root
  '
echo "::endgroup::"
[ -s "$WORK/runs/0/sim_with_tracks.root" ] || { echo "FAIL: digi_and_reco did not write sim_with_tracks.root"; exit 1; }

# ---------------------------------------------------------------------------
# Phase 2 (RECO image): Pandora charged PF on the ActsTracks file.
# ---------------------------------------------------------------------------
echo "::group::reco image — pandora_reco (charged PF on ActsTracks)"
docker run --rm -v "$REPO":/colliderml:ro -v "$WORK":/output \
  --entrypoint bash "$RECO_IMAGE" -c '
    set -euo pipefail
    source /opt/key4hep-shim.sh
    source /opt/pandora-stack/install/setup_stack.sh
    for s in /opt/odd-install/bin/this*ODD*.sh /opt/odd-install/bin/this_odd.sh; do [ -f "$s" ] && source "$s" && break; done
    export K4ODD_PATH=/opt/k4ODD ODD_INSTALL_DIR=/opt/odd-install
    export LD_LIBRARY_PATH=/opt/k4ODD/install/lib:/opt/k4ODD/install/lib64:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/opt/k4ODD/install/python:${PYTHONPATH:-}
    cd /colliderml/scripts/simulation
    python3 pandora_reco.py --config /output/pandora.yaml \
        --input-file /output/runs/0/sim_with_tracks.root --output /output/runs/0
    podio-dump /output/runs/0/sim_with_tracks.root | grep -q ActsTracks || { echo "no ActsTracks in handoff file"; exit 1; }
    podio-dump /output/runs/0/reco_edm4hep.root | tee /output/runs/0/reco_dump.txt
  '
echo "::endgroup::"

# ---------------------------------------------------------------------------
# Assert charged PF came out of the ACTS-tracked file.
# ---------------------------------------------------------------------------
[ -s "$WORK/runs/0/reco_edm4hep.root" ] || { echo "FAIL: pandora_reco did not write reco_edm4hep.root"; exit 1; }
rc=0
for coll in GaudiPandoraPFOs GaudiPandoraClusters; do
  grep -q "$coll" "$WORK/runs/0/reco_dump.txt" && echo "  OK  $coll" || { echo "  FAIL missing $coll"; rc=1; }
done
[ "$rc" -eq 0 ] && echo "CHARGED-PF TWO-CONTAINER E2E PASSED" || { echo "E2E FAILED"; exit 1; }
