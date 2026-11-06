#!/bin/bash
# Submit the v22 PU200 ttbar extension (to 100k events) as regular multi-node jobs on account
# m5386 through scripts/cli/run_blocks_multinode.sh.
#   v22_submit.sh <target> [extra sbatch args]
# Targets: ttbar_pu200_trk_test, ttbar_pu200_trk
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
S=/global/cfs/cdirs/m4958/data/ColliderML/simulation
ST=/global/cfs/cdirs/m4958/data/ColliderML/staging/simulation
DRIVER="$REPO/scripts/cli/run_blocks_multinode.sh"
TRK=digitization_config.yaml

submit() {  # name nodes time logdir config first last per_wave [extra]
  local name=$1 nodes=$2 time=$3 logdir=$4 cfg=$5 first=$6 last=$7 wave=$8; shift 8
  mkdir -p "$logdir"
  sbatch -A m5386 -C cpu -q regular -N "$nodes" -t "$time" -J "$name" -o "$logdir/${name}_%j.out" "$@" \
    --wrap "bash $DRIVER $cfg $first $last $wave"
}

case $1 in
  ttbar_pu200_trk_test)  # 2 nodes, runs 157-158 (one each): checks multi-node launch on regular
    submit ttbar-pu200-trk-test 2 00:45:00 $S/full_pileup/ttbar/v22/logs/blocks \
      "$REPO/configs_production/full_pileup/ttbar/v22/$TRK" 157 158 1 "${@:2}" ;;
  ttbar_pu200_trk)       # runs 159-1562: 130 nodes x 11 runs, one wave (~16 min at 16 per node in v22)
    submit ttbar-pu200-trk 130 01:00:00 $S/full_pileup/ttbar/v22/logs/blocks \
      "$REPO/configs_production/full_pileup/ttbar/v22/$TRK" 159 1562 11 "${@:2}" ;;
  gen_pu0)               # generator_particles, PU0 runs 0-7 on 1 node
    mkdir -p $S/hard_scatter/ttbar/v22/logs/generator_particles
    sbatch -A m5386 -C cpu -q regular -N1 --ntasks 8 --cpus-per-task 16 -t 00:30:00 -J gen-pu0 \
      -o $S/hard_scatter/ttbar/v22/logs/generator_particles/job_%j.out "${@:2}" \
      "$REPO/scripts/production/generator_particles.sbatch" \
      "$REPO/configs_production/hard_scatter/ttbar/v22/generator_particles_config.yaml" 0 ;;
  gen_pu200)             # generator_particles, PU200 runs 0-1562: 98 nodes x 16 runs
    mkdir -p $S/full_pileup/ttbar/v22/logs/generator_particles
    sbatch -A m5386 -C cpu -q regular -N98 --ntasks 1563 --cpus-per-task 16 -t 01:00:00 -J gen-pu200 \
      -o $S/full_pileup/ttbar/v22/logs/generator_particles/job_%j.out "${@:2}" \
      "$REPO/scripts/production/generator_particles.sbatch" \
      "$REPO/configs_production/full_pileup/ttbar/v22/generator_particles_config.yaml" 0 ;;
  pkg_gen_pu0)           # package generator_particles, PU0 chunks 0-10 on 1 node
    submit pkg-gen-pu0 1 00:30:00 $S/hard_scatter/ttbar/v22/logs/blocks \
      "$REPO/configs_production/hard_scatter/ttbar/v22/package_generator_config.yaml" 0 10 11 "${@:2}" ;;
  # Pandora rerun with reco 20261106 into staging (configs point output_base_dir at staging)
  pandora_pu200_a)       # runs 0-156 (tracked): 157 nodes, one run per node (~45 min)
    submit pandora-pu200-a 157 01:30:00 $ST/full_pileup/ttbar/v22/logs/blocks \
      "$REPO/configs_production/full_pileup/ttbar/v22/pandora_reco_config.yaml" 0 156 1 "${@:2}" ;;
  pandora_pu200_b)       # runs 157-1562: 201 nodes x 7 runs one after another (~5.5 h); add --dependency
    submit pandora-pu200-b 201 07:00:00 $ST/full_pileup/ttbar/v22/logs/blocks \
      "$REPO/configs_production/full_pileup/ttbar/v22/pandora_reco_config.yaml" 157 1562 1 "${@:2}" ;;
  pandora_pu0)           # runs 0-7 on 1 node, all 8 at once (16 slices each)
    submit pandora-pu0 1 01:00:00 $ST/hard_scatter/ttbar/v22/logs/blocks \
      "$REPO/configs_production/hard_scatter/ttbar/v22/pandora_reco_config.yaml" 0 7 8 "${@:2}" ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
