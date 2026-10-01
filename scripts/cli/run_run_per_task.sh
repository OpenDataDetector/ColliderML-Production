#!/bin/bash
# Spread numbered runs over the SLURM tasks (nodes) of one interactive allocation:
# task k processes runs FIRST + k*PER .. FIRST + k*PER + PER-1 (capped at LAST), in
# parallel on its node via run_runs_interactive.sh.
#   salloc -N 4 ... srun -N4 -n4 -c256 scripts/cli/run_run_per_task.sh <config.yaml> <first> <last> [per]
# PER defaults to 1 (one run per node). Tasks with no runs left exit 0.
CFG=$1; FIRST=$2; LAST=$3; PER=${4:-1}
START=$((FIRST + SLURM_PROCID * PER))
END=$((START + PER - 1)); [ "$END" -le "$LAST" ] || END=$LAST
[ "$START" -le "$LAST" ] || exit 0
exec "$(dirname "$0")/run_runs_interactive.sh" "$CFG" "$START" "$END"
