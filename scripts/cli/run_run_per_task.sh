#!/bin/bash
# One numbered run per SLURM task (node): run = FIRST + SLURM_PROCID. Lets one interactive
# allocation of N nodes process N runs of a stage, one run per node:
#   salloc -N 4 ... srun -N4 -n4 -c256 scripts/cli/run_run_per_task.sh <config.yaml> <first> <last>
# Tasks whose run would exceed LAST exit 0 without work.
CFG=$1; FIRST=$2; LAST=$3
RUN=$((FIRST + SLURM_PROCID))
[ "$RUN" -le "$LAST" ] || exit 0
exec "$(dirname "$0")/run_runs_interactive.sh" "$CFG" "$RUN" "$RUN"
