#!/bin/bash
# Run one stage config over runs FIRST..LAST inside a multi-node Slurm allocation: the runs are
# split into contiguous blocks, one block per node, and each node runs its block in waves of
# PER_WAVE runs in parallel (run_runs_interactive.sh, one run_stage process per run).
#
#   sbatch -A m5386 -C cpu -q regular -N 130 -t 01:00:00 --wrap \
#     "bash scripts/cli/run_blocks_multinode.sh <config.yaml> <first> <last> <per_wave>"
#   ... run_blocks_multinode.sh <config.yaml> --units <file with one unit id per line> <per_wave>
#       (production: the unit list written by scripts/production/v23_control.py plan; only enumerated
#        missing or failed units, never a range that covers accepted ones)
#
# Per-node logs: <version dir>/logs/blocks/<stage>_<job>_<node>.log (run logs as in
# run_runs_interactive.sh). Runs whose stage already succeeded are not skipped: pass a range
# that covers only the runs to (re)do. Exit status: for exactly one run, that task's own code; otherwise the number of nodes with failed runs (<= 255).
set -uo pipefail
CFG=$(readlink -f "$1")
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/global/homes/d/danieltm/.conda/envs/collider-env/bin/python}
if [ "$2" = "--units" ]; then
  [ -f "$3" ] || { echo "unit list $3 not found"; exit 2; }
  mapfile -t UNITS < <(grep -v '^[[:space:]]*$' "$3")
  PER_WAVE=$4
else
  FIRST=$2; LAST=$3; PER_WAVE=$4
  [[ "$FIRST" =~ ^[0-9]+$ && "$LAST" =~ ^[0-9]+$ ]] && [ "$FIRST" -le "$LAST" ] \
    || { echo "invalid range: FIRST=$FIRST LAST=$LAST"; exit 2; }
  mapfile -t UNITS < <(seq "$FIRST" "$LAST")
fi
[[ "$PER_WAVE" =~ ^[1-9][0-9]*$ && ${#UNITS[@]} -gt 0 ]] || { echo "invalid wave size $PER_WAVE or no units"; exit 2; }
for u in "${UNITS[@]}"; do [[ "$u" =~ ^[0-9]+$ ]] || { echo "invalid unit '$u'"; exit 2; }; done
[ "$(printf '%s\n' "${UNITS[@]}" | sort -u | wc -l)" -eq "${#UNITS[@]}" ] || { echo "duplicate units"; exit 2; }
# provenance gate before any write or image load (exit 1, nothing created)
$PY "$REPO/scripts/cli/provenance_preflight.py" "$CFG" || exit 1
read -r STAGE VDIR < <($PY - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
base = c.get("common", {}).get("output_base_dir") or c.get("output_base_dir")
print(c["stage"], f"{base}/{c['campaign']}/{c['dataset']}/{c['version']}")
EOF
)
LOGDIR="$VDIR/logs/blocks"; mkdir -p "$LOGDIR"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N=${#NODES[@]}; TOTAL=${#UNITS[@]}; PER_NODE=$(( (TOTAL + N - 1) / N ))
echo "=== $(date '+%F %T') $STAGE ${TOTAL} units (${UNITS[0]}..${UNITS[-1]}) on $N nodes, $PER_NODE per node, $PER_WAVE per wave"

node_block() {  # node first-index last-index (into UNITS)
  local node=$1 a=$2 b=$3 fail=0
  for ((w = a; w <= b; w += PER_WAVE)); do
    local e=$((w + PER_WAVE - 1)); [ $e -gt $b ] && e=$b
    local wave; wave=$(IFS=,; echo "${UNITS[*]:w:e-w+1}")
    echo "=== $(date '+%F %T') $node wave $wave"
    srun -N1 -n1 -c256 -w "$node" --overlap bash "$REPO/scripts/cli/run_runs_interactive.sh" "$CFG" --units "$wave"
    local rc=$?; echo "=== $(date '+%F %T') $node wave $wave rc=$rc"
    # one-run wave: rc is that task's own code; several runs: rc is the number of failed runs
    [ $rc -eq 0 ] || { fail=$((fail + 1)); last=$rc; }
  done
  echo "=== $(date '+%F %T') $node done failed_waves=$fail"
  [ $fail -eq 0 ] && return 0
  return ${last:-1}
}

pids=()
for ((k = 0; k < N; k++)); do
  a=$((k * PER_NODE)); [ $a -ge $TOTAL ] && break
  b=$((a + PER_NODE - 1)); [ $b -ge $TOTAL ] && b=$((TOTAL - 1))
  node_block "${NODES[$k]}" "$a" "$b" > "$LOGDIR/${STAGE}_${SLURM_JOB_ID}_${NODES[$k]}.log" 2>&1 &
  pids+=($!)
done
bad=0; last=0
for p in "${pids[@]}"; do wait "$p"; rc=$?; [ $rc -eq 0 ] || { bad=$((bad + 1)); last=$rc; }; done
echo "=== $(date '+%F %T') $STAGE done: $bad node(s) with failed runs"
grep -h "rc=[1-9]" "$VDIR"/logs/blocks/"${STAGE}"_"${SLURM_JOB_ID}"_*.log 2>/dev/null | head -20
# exactly one run in total: exit with that task's code; otherwise the number of failed nodes (<= 255)
if [ "$TOTAL" -eq 1 ]; then exit $last; fi
exit $(( bad > 255 ? 255 : bad ))
