#!/bin/bash
# Run one stage config over several numbered runs IN PARALLEL inside an interactive
# allocation: one `run_stage.py <config> --execution-mode interactive --run-list N` per
# run, i.e. the same arguments a batch task gets. Use for pilots, which must not go to
# the regular/shared queues.
#
#   salloc -A m4958 -C cpu -q interactive -t 120 -N 1 \
#       srun -N1 -n1 -c256 bash scripts/cli/run_runs_interactive.sh <config.yaml> <first> <last>
#
# Per-run logs: <version dir>/logs/interactive/<stage>_run<N>.log. Exit status is the
# number of failed runs (0 = all good). If the config names a container_tarball for this
# stage and the image is not in the podman store, it is loaded first (batch jobs do this
# in job_submission.py; interactive run_stage does not).
set -uo pipefail
CFG=$(readlink -f "$1"); FIRST=$2; LAST=$3
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/global/homes/d/danieltm/.conda/envs/collider-env/bin/python}

read -r STAGE VERSION_DIR IMAGE TARBALL < <($PY - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
stage = c["stage"]
base = c.get("common", {}).get("output_base_dir") or c.get("output_base_dir")
common = c.get("common", {})
sc = common.get("stage_containers", {}).get(stage, {})
# Stage override first, else the config's default container (as run_stage resolves it).
image = sc.get("container") or common.get("container") or "-"
tarball = sc.get("container_tarball") if sc.get("container") else common.get("container_tarball")
print(stage, f"{base}/{c['campaign']}/{c['dataset']}/{c['version']}", image, tarball or "-")
EOF
)
LOGDIR="$VERSION_DIR/logs/interactive"; mkdir -p "$LOGDIR"
echo "=== $(date '+%F %T') $STAGE runs $FIRST..$LAST on $(hostname) ($(nproc) cpus)"

# `podman-hpc image exists` does not see migrated (squashed, read-only) images on compute
# nodes, so test by running the image instead.
if [ "$IMAGE" != "-" ] && ! podman-hpc run --rm --entrypoint /bin/true "$IMAGE" 2>/dev/null; then
  echo "=== loading $IMAGE from $TARBALL"
  podman-hpc load -i "$TARBALL" && podman-hpc image exists "$IMAGE" || { echo "image load failed"; exit 99; }
fi

cd "$REPO"
declare -A PIDS
for n in $(seq "$FIRST" "$LAST"); do
  $PY scripts/cli/run_stage.py "$CFG" --execution-mode interactive --run-list "$n" \
      > "$LOGDIR/${STAGE}_run${n}.log" 2>&1 &
  PIDS[$n]=$!
done
fail=0
for n in $(seq "$FIRST" "$LAST"); do
  wait "${PIDS[$n]}"; rc=$?
  echo "=== $(date '+%F %T') $STAGE run $n rc=$rc"
  [ $rc -eq 0 ] || fail=$((fail + 1))
done
echo "=== $(date '+%F %T') $STAGE done: $fail failed"
exit $fail
