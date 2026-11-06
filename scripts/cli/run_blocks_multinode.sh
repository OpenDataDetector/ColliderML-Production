#!/bin/bash
# Run one stage config over runs FIRST..LAST inside a multi-node Slurm allocation: the runs are
# split into contiguous blocks, one block per node, and each node runs its block in waves of
# PER_WAVE runs in parallel (run_runs_interactive.sh, one run_stage process per run).
#
#   sbatch -A m5386 -C cpu -q regular -N 130 -t 01:00:00 --wrap \
#     "bash scripts/cli/run_blocks_multinode.sh <config.yaml> <first> <last> <per_wave>"
#
# Per-node logs: <version dir>/logs/blocks/<stage>_<job>_<node>.log (run logs as in
# run_runs_interactive.sh). Runs whose stage already succeeded are not skipped: pass a range
# that covers only the runs to (re)do. Exit status: number of nodes with failed runs.
set -uo pipefail
CFG=$(readlink -f "$1"); FIRST=$2; LAST=$3; PER_WAVE=$4
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/global/homes/d/danieltm/.conda/envs/collider-env/bin/python}
read -r STAGE VDIR < <($PY - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
base = c.get("common", {}).get("output_base_dir") or c.get("output_base_dir")
print(c["stage"], f"{base}/{c['campaign']}/{c['dataset']}/{c['version']}")
EOF
)
LOGDIR="$VDIR/logs/blocks"; mkdir -p "$LOGDIR"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N=${#NODES[@]}; TOTAL=$((LAST - FIRST + 1)); PER_NODE=$(( (TOTAL + N - 1) / N ))
echo "=== $(date '+%F %T') $STAGE runs $FIRST..$LAST on $N nodes, $PER_NODE per node, $PER_WAVE per wave"

node_block() {  # node first last
  local node=$1 a=$2 b=$3 fail=0
  for ((w = a; w <= b; w += PER_WAVE)); do
    local e=$((w + PER_WAVE - 1)); [ $e -gt $b ] && e=$b
    echo "=== $(date '+%F %T') $node wave $w..$e"
    srun -N1 -n1 -c256 -w "$node" --overlap bash "$REPO/scripts/cli/run_runs_interactive.sh" "$CFG" "$w" "$e"
    local rc=$?; echo "=== $(date '+%F %T') $node wave $w..$e failed_runs=$rc"; fail=$((fail + rc))
  done
  echo "=== $(date '+%F %T') $node done failed_runs=$fail"
  return $((fail > 0))
}

pids=()
for ((k = 0; k < N; k++)); do
  a=$((FIRST + k * PER_NODE)); [ $a -gt $LAST ] && break
  b=$((a + PER_NODE - 1)); [ $b -gt $LAST ] && b=$LAST
  node_block "${NODES[$k]}" "$a" "$b" > "$LOGDIR/${STAGE}_${SLURM_JOB_ID}_${NODES[$k]}.log" 2>&1 &
  pids+=($!)
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=$((bad + 1)); done
echo "=== $(date '+%F %T') $STAGE done: $bad node(s) with failed runs"
grep -h "rc=[1-9]" "$VDIR"/logs/blocks/"${STAGE}"_"${SLURM_JOB_ID}"_*.log 2>/dev/null | head -20
exit $bad
