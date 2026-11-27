#!/bin/bash
# Run a validation task list in the earlier image reco:20261106, which has neither the
# colliderml-reco command nor env/setup.sh. Optionally replace its Pandora install prefix,
# to test another build of the Pandora libraries in an otherwise unchanged image.
#
# Usage: run_validation_reference_image.sh TASKS.json OUT.json [PANDORA_INSTALL_DIR [SRC:DST...]]
#   WORKERS (environment, default 64) is the number of parallel processes.
#   PANDORA_INSTALL_DIR may be "" to keep the image's own libraries; SRC:DST are extra
#   read-only mounts (for task lists whose references are not on CFS).
set -euo pipefail

tasks="$(realpath "${1:?usage: run_validation_reference_image.sh TASKS.json OUT.json [PANDORA_INSTALL_DIR]}")"
out="$(realpath -m "${2:?missing OUT.json}")"
install="${3:-}"
extra_mounts=("${@:4}")
image=localhost/colliderml/reco:20261106
recipe="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cfs=/global/cfs/cdirs/m4958
work="$(mktemp -d /tmp/reco-validation-reference.XXXXXX)"

python3 - "$tasks" "$work/config.json" "${WORKERS:-64}" <<'PY'
import json, sys
tasks = json.load(open(sys.argv[1]))
json.dump({"tasks": tasks, "work_dir": "/validation-work", "workers": int(sys.argv[3]), "first_core": 0,
           "env": [],
           "k4run_args": [],
           "keep_outputs": False},
          open(sys.argv[2], "w"))
PY

mounts=()
if [[ -n "$install" ]]; then
  # Gaudi opens the *.confdb2 files of the prefix read-write, so mount a writable copy.
  cp -a "$install" "$work/install"
  mounts=(-v "$work/install:/opt/pandora-stack/install")
fi
for mount in "${extra_mounts[@]}"; do
  mounts+=(-v "$mount:ro")
done

podman-hpc run --rm --network=none "${mounts[@]}" \
  -v "$recipe/bin/colliderml-reco:/usr/local/bin/colliderml-reco:ro" \
  -v "$recipe/validation:/validation:ro" -v "$work:/validation-work" -v "$cfs:$cfs:ro" \
  "$image" bash -c '
    source /opt/key4hep-shim.sh
    source /opt/pandora-stack/install/setup_stack.sh
    source /opt/odd-install/bin/this_odd.sh
    export K4ODD_PATH=/opt/k4ODD ODD_INSTALL_DIR=/opt/odd-install
    export LD_LIBRARY_PATH=/opt/k4ODD/install/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/k4ODD/install/python:$PYTHONPATH
    python3 /validation/run_validation.py --inner /validation-work/config.json'

python3 - "$work/records.json" <<'PY'
import json, sys
failed = [r["name"] for r in json.load(open(sys.argv[1])) if "error" in r]
if failed:
    sys.exit(f"runs or comparisons did not complete: {failed}")
PY
cp "$work/records.json" "$out"
