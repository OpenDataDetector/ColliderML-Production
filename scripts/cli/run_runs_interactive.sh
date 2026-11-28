#!/bin/bash
# Run one stage config over several numbered runs IN PARALLEL inside an interactive
# allocation: one `run_stage.py <config> --execution-mode interactive --run-list N` per
# run, i.e. the same arguments a batch task gets. Use for pilots, which must not go to
# the regular/shared queues.
#
#   salloc -A m4958 -C cpu -q interactive -t 120 -N 1 \
#       srun -N1 -n1 -c256 bash scripts/cli/run_runs_interactive.sh <config.yaml> <first> <last>
#   ... run_runs_interactive.sh <config.yaml> --units 3,17,42      (explicit units, e.g. enumerated failures)
#
# Per-attempt logs: <version dir>/logs/interactive/<stage>_run<N>.<job>.<step>.<host>.<pid>.log (a retry never
# truncates an earlier attempt's log; run_stage takes the unit lock before any other write). Exit status: for one run,
# that run's own exit code; for several, the number of failed runs (0 = all good). If the config names a container_tarball for this
# stage and the image is not in the podman store, it is loaded first (batch jobs do this
# in job_submission.py; interactive run_stage does not).
set -uo pipefail
CFG=$(readlink -f "$1")
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/global/homes/d/danieltm/.conda/envs/collider-env/bin/python}
if [ "${2:-}" = "--units" ]; then
  [[ "${3:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid unit list '${3:-}'"; exit 2; }
  IFS=, read -r -a UNITS <<< "$3"
else
  FIRST=${2:-}; LAST=${3:-}
  [[ "$FIRST" =~ ^[0-9]+$ && "$LAST" =~ ^[0-9]+$ ]] && [ "$FIRST" -le "$LAST" ] \
    || { echo "invalid run range FIRST=$FIRST LAST=$LAST"; exit 2; }
  mapfile -t UNITS < <(seq "$FIRST" "$LAST")
fi
[ "$(printf '%s\n' "${UNITS[@]}" | sort -u | wc -l)" -eq "${#UNITS[@]}" ] || { echo "duplicate units in ${UNITS[*]}"; exit 2; }
# provenance gate before any write or image load (exit 1, nothing created)
$PY "$REPO/scripts/cli/provenance_preflight.py" "$CFG" || exit 1

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
echo "=== $(date '+%F %T') $STAGE units ${UNITS[*]} on $(hostname) ($(nproc) cpus)"
TOKEN="${SLURM_JOB_ID:-nojob}.${SLURM_STEP_ID:-nostep}.$(hostname -s).$$"

# `podman-hpc image exists` does not see migrated (squashed, read-only) images on compute
# nodes, so test by running the image instead.
if [ "$IMAGE" != "-" ] && ! podman-hpc run --rm --entrypoint /bin/true "$IMAGE" 2>/dev/null; then
  echo "=== loading $IMAGE from $TARBALL"
  podman-hpc load -i "$TARBALL" && podman-hpc image exists "$IMAGE" || { echo "image load failed"; exit 99; }
fi

# a dispatched block (v23_control.py dispatch exports the frozen launch record): this node repeats the identity
# check (code, configs, images on THIS node, archive) and, per unit of the wave, the rule dispatch applied (state,
# failure count, upstream accepted and bound), before anything is written
if [ -n "${COLLIDERML_LAUNCH_RECORD:-}" ]; then
  $PY "$REPO/scripts/production/v23_control.py" unit-check "$COLLIDERML_LAUNCH_RECORD" --config "$CFG" \
      --units "$(IFS=,; echo "${UNITS[*]}")" || { echo "unit-check refused the wave ${UNITS[*]} on $(hostname)"; exit 1; }
fi
LOGDIR="$VERSION_DIR/logs/interactive"; mkdir -p "$LOGDIR"
cd "$REPO"
declare -A PIDS
for n in "${UNITS[@]}"; do
  $PY scripts/cli/run_stage.py "$CFG" --execution-mode interactive --run-list "$n" \
      > "$LOGDIR/${STAGE}_run${n}.${TOKEN}.log" 2>&1 &
  PIDS[$n]=$!
done
fail=0; last_rc=0
for n in "${UNITS[@]}"; do
  wait "${PIDS[$n]}"; rc=$?
  echo "=== $(date '+%F %T') $STAGE run $n rc=$rc"
  [ $rc -eq 0 ] || { fail=$((fail + 1)); last_rc=$rc; }
done
echo "=== $(date '+%F %T') $STAGE done: $fail failed"
# One run: exit with that task's own code (e.g. 137 stays 137). Several runs: the number of
# failed runs (capped at 255 by the shell), every run's code is in the lines above.
if [ "${#UNITS[@]}" -eq 1 ]; then exit $last_rc; fi
exit $(( fail > 255 ? 255 : fail ))
