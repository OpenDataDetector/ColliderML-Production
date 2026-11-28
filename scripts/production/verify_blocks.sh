#!/bin/bash
# Verification block (v23_control.py dispatch --stage verify): the sealed chunk list split into contiguous blocks,
# one per node, each node verifying PER_WAVE chunks at a time (v23_control.py verify-chunk, which refuses a chunk
# that is not accepted and bound or already has a current passing report, and moves an earlier report aside).
#   verify_blocks.sh <launch record> <campaign> <unit file> <per_wave>
# Every child's exit code is collected by its own pid (no bare wait); exit status = number of failed chunks (<= 255).
set -uo pipefail
REC=$1; CAMP=$2; UF=$3; PER_WAVE=$4
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/global/homes/d/danieltm/.conda/envs/collider-env/bin/python}
[ -f "$UF" ] || { echo "unit list $UF not found"; exit 2; }
if [ -n "${COLLIDERML_UNITS_SHA256:-}" ] && [ "$(sha256sum < "$UF" | cut -d' ' -f1)" != "$COLLIDERML_UNITS_SHA256" ]; then
  echo "unit list $UF differs from the dispatched block (sha256)"; exit 2
fi
mapfile -t UNITS < <(grep -v '^[[:space:]]*$' "$UF")
[[ "$PER_WAVE" =~ ^[1-9][0-9]*$ && ${#UNITS[@]} -gt 0 ]] || { echo "invalid wave size $PER_WAVE or no units"; exit 2; }
for u in "${UNITS[@]}"; do [[ "$u" =~ ^[0-9]+$ ]] || { echo "invalid unit '$u'"; exit 2; }; done
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N=${#NODES[@]}; TOTAL=${#UNITS[@]}; PER_NODE=$(( (TOTAL + N - 1) / N ))
TMPD=${TMPDIR:-/tmp}
echo "=== $(date '+%F %T') verify $CAMP ${TOTAL} chunks on $N nodes, $PER_WAVE per wave"
node_block() {  # node a b
  local node=$1 a=$2 b=$3 fail=0
  for ((w = a; w <= b; w += PER_WAVE)); do
    local e=$((w + PER_WAVE - 1)); [ $e -gt $b ] && e=$b
    local wave="${UNITS[*]:w:e-w+1}"
    echo "=== $(date '+%F %T') $node wave $wave"
    srun -N1 -n1 -c256 -w "$node" --overlap bash -c '
      PY=$1; REPO=$2; REC=$3; CAMP=$4; T=$5; shift 5; declare -A P; bad=0
      for k in "$@"; do $PY $REPO/scripts/production/v23_control.py verify-chunk $REC --campaign $CAMP --chunk $k --tmpdir $T & P[$k]=$!; done
      for k in "$@"; do wait ${P[$k]}; rc=$?; echo "chunk $k rc=$rc"; [ $rc -eq 0 ] || bad=$((bad + 1)); done
      exit $bad' _ "$PY" "$REPO" "$REC" "$CAMP" "$TMPD" $wave
    local rc=$?; echo "=== $(date '+%F %T') $node wave rc=$rc"
    fail=$((fail + rc))
  done
  return $(( fail > 255 ? 255 : fail ))
}
pids=()
for ((k = 0; k < N; k++)); do
  a=$((k * PER_NODE)); [ $a -ge $TOTAL ] && break
  b=$((a + PER_NODE - 1)); [ $b -ge $TOTAL ] && b=$((TOTAL - 1))
  node_block "${NODES[$k]}" "$a" "$b" & pids+=($!)
done
bad=0
for p in "${pids[@]}"; do wait "$p"; rc=$?; bad=$((bad + rc)); done
echo "=== $(date '+%F %T') verify $CAMP done: $bad failed chunk(s)"
exit $(( bad > 255 ? 255 : bad ))
