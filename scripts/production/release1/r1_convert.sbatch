#!/bin/bash
# Release 1 packaging: convert_all.py chunks FIRST..FIRST+ntasks-1 (one srun task per chunk) in
# collider-env as in Release 1, with the issue #53 converter (release1_code/58c541d: a short calo
# chunk raises), then validate_calo_hits.py on every calo_hits file written.
#
#   sbatch -A m5386 -C cpu -q regular -N <nodes> --ntasks <chunks> --ntasks-per-node 3 \
#     --cpus-per-task 85 -t 02:00:00 r1_convert.sbatch <config.yaml> <first_chunk>
CFG=$1; FIRST=$2
CODE=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/release1_code/58c541d/scripts
PY=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
export SLURM_CPU_BIND="cores"
LAST=$((FIRST + SLURM_NTASKS - 1))
read -r V < <($PY - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(f"{c['common']['output_base_dir']}/{c['campaign']}/{c['dataset']}/{c['version']}")
EOF
)
OUT=$V/parquet/reco/calo_hits; LOGD=$V/logs/r1_convert; mkdir -p "$LOGD"
echo "=== $(date '+%F %T') convert_all chunks $FIRST..$LAST on $SLURM_JOB_NUM_NODES nodes (job $SLURM_JOB_ID)"
srun --exact --kill-on-bad-exit=0 -u bash -c "N=\$(($FIRST + SLURM_PROCID)); \
  $PY $CODE/postprocessing/convert_all.py --config $CFG --chunk-index \$N > $LOGD/chunk_\$N.log 2>&1; \
  echo \"=== chunk \$N rc=\$?\""
files=()
for ((n = FIRST; n <= LAST; n++)); do
  f=$(ls "$OUT"/*.events$((n * 1000))-$((n * 1000 + 999)).parquet 2>/dev/null); [ -n "$f" ] && files+=("$f")
done
echo "=== $(date '+%F %T') validating ${#files[@]} calo_hits files"
$PY $CODE/dataset/validate_calo_hits.py --expected-events 1000 "${files[@]}"
echo "=== $(date '+%F %T') done validate_rc=$?"
