#!/bin/bash
# Submit Release 1 format jobs for the multijet (jets) and diphoton PU0 samples (account m5386).
#   r1_submit.sh <target> [extra sbatch args]
# Tests (small, before the large runs are approved):
#   jets_convert_test      chunks 0-2 on 1 node
#   diphoton_sim_test      Geant4 runs 0-1 on 1 node
#   diphoton_trk_test      tracking runs 0-1 on 1 node (after diphoton_sim_test)
#   diphoton_convert_test  chunks 0-1 on 1 node (after diphoton_trk_test)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$REPO/scripts/production/release1"
S=/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter
IMG=(--image registry.cern.ch/atlasadc/atlas-grid-almalinux9 --module=cvmfs)
SB=(sbatch -A m5386 -C cpu -q regular)
SIM=simulation_config.yaml; TRK=digitization_config.yaml; CONV=convert_all.yaml

case $1 in
  jets_convert_test)
    mkdir -p $S/jets/v1/logs/r1_convert
    "${SB[@]}" -N1 --ntasks 3 --ntasks-per-node 3 --cpus-per-task 85 -t 02:00:00 -J jets-conv-test \
      -o $S/jets/v1/logs/r1_convert/test_%j.out "${@:2}" "$D/r1_convert.sbatch" "$REPO/configs_production/hard_scatter/jets/v1/$CONV" 0 ;;
  diphoton_sim_test)
    mkdir -p $S/diphoton/v1/logs/r1_simulation
    "${SB[@]}" -N1 --ntasks 2 --cpus-per-task 8 -t 03:00:00 -J dipho-sim-test "${IMG[@]}" \
      -o $S/diphoton/v1/logs/r1_simulation/test_%j.out "${@:2}" "$D/r1_stage.sbatch" ddsim_run \
      "$REPO/configs_production/hard_scatter/diphoton/v1/$SIM" $S/diphoton/v1/runs diphoton 0 ;;
  diphoton_trk_test)
    mkdir -p $S/diphoton/v1/logs/r1_tracking
    "${SB[@]}" -N1 --ntasks 2 --cpus-per-task 16 -t 01:00:00 -J dipho-trk-test "${IMG[@]}" \
      -o $S/diphoton/v1/logs/r1_tracking/test_%j.out "${@:2}" "$D/r1_stage.sbatch" digi_and_reco \
      "$REPO/configs_production/hard_scatter/diphoton/v1/$TRK" $S/diphoton/v1/runs diphoton 0 ;;
  diphoton_convert_test)
    mkdir -p $S/diphoton/v1/logs/r1_convert
    "${SB[@]}" -N1 --ntasks 2 --ntasks-per-node 3 --cpus-per-task 85 -t 02:00:00 -J dipho-conv-test \
      -o $S/diphoton/v1/logs/r1_convert/test_%j.out "${@:2}" "$D/r1_convert.sbatch" "$REPO/configs_production/hard_scatter/diphoton/v1/$CONV" 0 ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
