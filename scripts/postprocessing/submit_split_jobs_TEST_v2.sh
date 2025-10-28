#!/bin/bash
#SBATCH --job-name=split_test
#SBATCH --output=logs/split_test_%j.out
#SBATCH --error=logs/split_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4

# TEST version: Single SLURM job with multiple nodes for parallel HepMC file splitting
# Testing on small 16-run dataset - 4 nodes, each processing 4 runs with 4 processes

# ==============================================================================
# TEST CONFIGURATION
# ==============================================================================

# Path to the TEST runs directory
RUNS_DIR="/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/postprocessing/testing/runs"

# Offset for new run directories
OFFSET=16

# Events per split
EVENTS_PER_SPLIT=64

# Path to the Python script
SCRIPT_DIR="/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/postprocessing"
PYTHON_SCRIPT="${SCRIPT_DIR}/batch_split_runs.py"

# ==============================================================================
# END CONFIGURATION
# ==============================================================================

# Total runs and runs per task for automatic calculation
TOTAL_RUNS=16
RUNS_PER_TASK=4

export SLURM_CPU_BIND="cores"
srun --exact --kill-on-bad-exit=1 -u bash -c "
TASK_ID=\$SLURM_PROCID
MIN_RUN=\$((TASK_ID * ${RUNS_PER_TASK}))
MAX_RUN=\$((MIN_RUN + ${RUNS_PER_TASK} - 1))

# Don't exceed total runs
if [ \$MAX_RUN -ge ${TOTAL_RUNS} ]; then
    MAX_RUN=\$((${TOTAL_RUNS} - 1))
fi

echo '========================================================================'
echo \"Task \$TASK_ID on \$(hostname) [TEST MODE]\"
echo '========================================================================'
echo \"Job ID: ${SLURM_JOB_ID}\"
echo \"Processing runs: \$MIN_RUN to \$MAX_RUN\"
echo \"Time started: \$(date)\"
echo '========================================================================'

python ${PYTHON_SCRIPT} \\
    ${RUNS_DIR} \\
    -N ${OFFSET} \\
    -n ${EVENTS_PER_SPLIT} \\
    --min-run \$MIN_RUN \\
    --max-run \$MAX_RUN \\
    -j 4 \\
    -v

EXIT_CODE=\$?

echo '========================================================================'
echo \"Task \$TASK_ID completed with exit code: \$EXIT_CODE\"
echo \"Time finished: \$(date)\"
echo '========================================================================'

exit \$EXIT_CODE
"
