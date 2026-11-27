#!/bin/bash
# Run the release validation on one Perlmutter CPU node (128 cores, 512 GB).
# Usage, inside an allocation: srun -N1 -n1 nersc/run_validation_perlmutter.sh IMAGE [SET...]
#   SET is a task list in nersc/tasks: suite345 pu200_192 pu0_run0 timing differing
#   reference_variant (default: the first three). Results go to validation/results/<tag>/.
set -euo pipefail

image="${1:?usage: run_validation_perlmutter.sh IMAGE [SET...]}"
shift
sets=("$@")
[[ ${#sets[@]} -gt 0 ]] || sets=(suite345 pu200_192 pu0_run0)

recipe="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validation="$recipe/validation"
tag="${image##*:}"
mkdir -p "$validation/results/$tag"
status=0
cfs=/global/cfs/cdirs/m4958
scratch="$(mktemp -d /tmp/reco-validation.XXXXXX)"

for name in "${sets[@]}"; do
  workers=64
  extra=()
  if [[ "$name" == timing ]]; then
    workers=1
    extra=(--env PANDORA_ALG_TIMING=1)
  fi
  python3 "$validation/run_validation.py" \
    --engine podman-hpc --image "$image" \
    --tasks "$recipe/nersc/tasks/$name.json" \
    --out "$validation/results/$tag/$name.json" \
    --work-dir "$scratch/$name" --workers "$workers" \
    --mount "$cfs:$cfs" "${extra[@]}" || {
      code=$?
      echo "$name: exit code $code (1: differences, 2: errors), see results/$tag/$name.json" >&2
      [[ $code -gt $status ]] && status=$code
    }
done
exit "$status"
