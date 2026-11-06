#!/bin/bash
# Submit the v22 production jobs (PU200 ttbar to 100k, multijet PU0 to 1M) as regular
# multi-node jobs on account m5386 through scripts/cli/run_blocks_multinode.sh.
#   v22_submit.sh <target> [extra sbatch args]
# Targets: jets_trk_test, jets_trk, ttbar_pu200_trk
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
S=/global/cfs/cdirs/m4958/data/ColliderML/simulation
DRIVER="$REPO/scripts/cli/run_blocks_multinode.sh"
TRK=digitization_config.yaml

submit() {  # name nodes time logdir config first last per_wave [extra]
  local name=$1 nodes=$2 time=$3 logdir=$4 cfg=$5 first=$6 last=$7 wave=$8; shift 8
  mkdir -p "$logdir"
  sbatch -A m5386 -C cpu -q regular -N "$nodes" -t "$time" -J "$name" -o "$logdir/${name}_%j.out" "$@" \
    --wrap "bash $DRIVER $cfg $first $last $wave"
}

case $1 in
  jets_trk_test)   # 2 nodes, one run each: checks multi-node launch on regular
    submit jets-trk-test 2 00:30:00 $S/hard_scatter/jets/v22/logs/blocks \
      "$REPO/configs_production/hard_scatter/jets/v22/$TRK" 10 11 1 "${@:2}" ;;
  jets_trk)        # runs 12-999: 132 nodes x 8 runs, one wave (~15 min)
    submit jets-trk 132 01:00:00 $S/hard_scatter/jets/v22/logs/blocks \
      "$REPO/configs_production/hard_scatter/jets/v22/$TRK" 12 999 8 "${@:2}" ;;
  ttbar_pu200_trk) # runs 157-1562: 130 nodes x 11 runs, one wave (~16 min at 16 per node in v22)
    submit ttbar-pu200-trk 130 01:00:00 $S/full_pileup/ttbar/v22/logs/blocks \
      "$REPO/configs_production/full_pileup/ttbar/v22/$TRK" 157 1562 11 "${@:2}" ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
