#!/bin/bash
# particles_full (generator_particles only) for runs FIRST..FIRST+ntasks-1, one srun task per run, in
# collider-env (pure python: uproot + scipy; no container needed). Logs: <version>/logs/generator_particles/.
#   sbatch -A m5386 -C cpu -q regular -N <nodes> --ntasks <runs> --cpus-per-task 8 -t 01:00:00 \
#     generator_particles.sbatch <config.yaml> <first>
CFG=$1; FIRST=$2
REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
[ -f "$REPO/scripts/postprocessing/convert_particles_full.py" ] || REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-prod-container/.claude/worktrees/release2-stages
PY=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
read -r V < <($PY - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(f"{c['common']['output_base_dir']}/{c['campaign']}/{c['dataset']}/{c['version']}")
EOF
)
LOGD=$V/logs/generator_particles; mkdir -p "$LOGD"
echo "=== $(date '+%F %T') generator_particles runs $FIRST..$((FIRST + SLURM_NTASKS - 1)) (job $SLURM_JOB_ID, code $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))"
srun --exact --kill-on-bad-exit=0 -u bash -c "N=\$(($FIRST + SLURM_PROCID)); \
  $PY $REPO/scripts/postprocessing/convert_particles_full.py --config $CFG --output $V/runs --output-subdir \$N \
  > $LOGD/run_\$N.log 2>&1; echo \"=== run \$N rc=\$?\""
echo "=== $(date '+%F %T') done"
