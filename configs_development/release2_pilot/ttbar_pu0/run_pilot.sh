#!/bin/bash
# Release-2 pilot on one Release-1 ttbar run (pileup 0): digitization -> pandora_reco -> reco_tables.
# Run INSIDE an interactive allocation, e.g.
#   salloc -A m4958 -C cpu -q interactive -t 120 -N 1 srun -N1 -n1 -c128 bash run_pilot.sh
# Every stage is driven by run_stage.py from this checkout so the configs are the record.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CFG="$REPO/configs_development/release2_pilot/ttbar_pu0"
PY=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
LOGDIR=/global/cfs/cdirs/m4958/data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/logs
mkdir -p "$LOGDIR"
cd "$REPO"

run_stage () {
  local name=$1 cfg=$2
  echo "=== $(date '+%F %T') START $name (host $(hostname))"
  local t0=$(date +%s)
  $PY scripts/cli/run_stage.py "$cfg" > "$LOGDIR/$name.log" 2>&1
  local rc=$? t1=$(date +%s)
  echo "=== $(date '+%F %T') END $name rc=$rc elapsed=$((t1 - t0))s"
  return $rc
}

run_stage digitization "$CFG/digitization_config.yaml" || { echo "digitization failed, stopping"; exit 1; }
run_stage pandora_reco "$CFG/pandora_reco_config.yaml" || { echo "pandora_reco failed, stopping"; exit 2; }
run_stage reco_tables "$CFG/reco_tables_config.yaml" || { echo "reco_tables failed"; exit 3; }
echo "=== pilot complete"
ls -la /global/cfs/cdirs/m4958/data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/runs/0/
