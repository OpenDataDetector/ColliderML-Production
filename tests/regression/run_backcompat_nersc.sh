#!/usr/bin/env bash
# Same-seed back-compat: convert_all (v1) vs native Arrow writer, on NERSC via
# podman-hpc + sw:pr-8. One digi+reco run emits BOTH ROOT and native parquet
# from the SAME seed-42 digitization -> exact comparison. Tracker-only muon
# input (particles/tracker_hits/tracks; no calo).
set -euo pipefail

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-prod-container
IMAGE="${IMAGE:-ghcr.io/opendatadetector/sw:pr-8}"
EDM="${EDM:-/global/cfs/cdirs/m4958/data/ColliderML/simulation/drift_beamspot/single_muon_10GeV/v1/runs/0/edm4hep.root}"
WORK="${WORK:-/pscratch/sd/d/danieltm/backcompat}"
SETUP=$REPO/scripts/cli/setup_container_env.sh
EVENTS=${EVENTS:-100}
mkdir -p "$WORK/runs/0"
# stale-output guard: a rerun (e.g. with smaller EVENTS) must not silently mix
# old parquet shards into the comparison — clear both trees first.
for d in "$WORK"/runs/0/particles "$WORK"/runs/0/tracker_hits \
         "$WORK"/runs/0/tracker_simhits "$WORK"/runs/0/tracks "$WORK"/backcompat; do
  [ -d "$d" ] && find "$d" -name '*.parquet' -delete
done

# --- digi config: seed 42, ROOT + native Arrow, tracker-only ODD v6.0.2 ---
cat >"$WORK/digi_config.yaml" <<EOF
campaign: "backcompat"
dataset: "muon"
version: "v1"
stage: "digitization"
seed: 42
events: ${EVENTS}
threads: 8
odd_geo_dir: /opt/odd
digi_config: /opt/odd/config/odd-digi-smearing-config.json
digi: true
reco: true
num_seeds_per_spm: 40
output_parquet_arrow: true
output_particles_root: true
output_measurements_root: true
ambi_root_output: true
ambi_finding_performance: true
ambi_fitting_performance: true
EOF

# --- convert_all config: tracker tables only ---
cat >"$WORK/convert_all.yaml" <<EOF
campaign: "backcompat"
dataset: "muon"
version: "v1"
stage: "convert_all"
input_base_dir: "$WORK"
common:
  output_base_dir: "$WORK"
objects:
  - tracker_hits
  - tracks
  - particles
output_format: "parquet"
chunk_size: 1000
run_size: ${EVENTS}
row_group_size: 100
particles_columns_keep: ["particle_id","pdg_id","mass","energy","charge","vx","vy","vz","px","py","pz","primary"]
digihits_columns_keep: ["x","y","z","particle_id"]
EOF

# source setup, then STRIP /cache from LD_LIBRARY_PATH so the setup-built ODD-v4
# factory lib doesn't shadow the /opt/odd v6 geometry factories (dd4hep plugin
# crash otherwise — the beamspot-refit gotcha).
run() { podman-hpc run --rm \
  -v /global/cfs/cdirs/m4958:/global/cfs/cdirs/m4958 -v /pscratch/sd/d/danieltm:/pscratch/sd/d/danieltm \
  --entrypoint /bin/bash "$IMAGE" -c "
    source $SETUP >$WORK/setup_container.log 2>&1   # persisted (container /tmp dies with --rm)
    export LD_LIBRARY_PATH=\$(printf '%s' \"\$LD_LIBRARY_PATH\" | tr ':' '\n' | grep -v '^/cache' | paste -sd:)
    $1"; }

echo "===== [1/3] digi+reco (seed 42, $EVENTS ev) -> ROOT + native Arrow ====="
run "cd $REPO/scripts/simulation
     python3 digi_and_reco.py --config $WORK/digi_config.yaml --input-file $EDM --output $WORK/runs/0"

# convert_all reads truth particles from edm4hep in the run dir
ln -sf "$EDM" "$WORK/runs/0/edm4hep.root"

echo "===== [2/3] convert_all.py on the ROOT -> v1 parquet ====="
run "cd $REPO/scripts/postprocessing
     python3 convert_all.py --config $WORK/convert_all.yaml"

echo "===== [3/3] inventory ====="
echo '--- native Arrow parquet (runs/0) ---'; find "$WORK/runs/0" -name '*.parquet' | sort | head
echo '--- v1 convert_all parquet ---'; find "$WORK/backcompat" -name '*.parquet' 2>/dev/null | sort | head
echo "DRIVER-DONE"
