#!/bin/bash
# Checks of the image that need no input data. Prints a log; exits non-zero on the first failure.
#
#   check_image.sh IMAGE [ENGINE]      ENGINE defaults to podman
set -euo pipefail

image="${1:?usage: check_image.sh IMAGE [ENGINE]}"
engine="${2:-podman}"
run=("$engine" run --rm --network=none)

echo "== image configuration (no setting of the build host)"
config="$("$engine" image inspect --format 'id: {{.Id}}
env: {{json .Config.Env}}
entrypoint: {{json .Config.Entrypoint}}
labels: {{json .Config.Labels}}
annotations: {{json .Annotations}}' "$image")"
echo "$config"
if grep -E "[Pp]odman|PODMAN|nersc|NERSC|/global/|/pscratch" <<< "$config"; then
  exit 1
fi

echo "== build record"
"${run[@]}" "$image" colliderml-reco info

echo "== self-test as root"
"${run[@]}" "$image" colliderml-reco selftest --events 2

echo "== self-test as an arbitrary user"
"${run[@]}" --user 12345:12345 "$image" colliderml-reco selftest --events 2

echo "== self-test as an arbitrary user on a read-only root file system"
"${run[@]}" --read-only --tmpfs /tmp --user 12345:12345 "$image" colliderml-reco selftest --events 2

echo "== NERSC paths and names in the recipe files, install prefixes and built libraries (expect none)"
"${run[@]}" "$image" bash -c '
  set -u
  # Case-sensitive: mangled C++ symbols contain strings such as "ERSC".
  pattern="/global/|/pscratch|nersc|NERSC|[Pp]erlmutter"
  grep -rIlE "$pattern" /opt/colliderml-reco /opt/odd-install /opt/k4reco-install \
      /opt/pandora-stack/install /opt/k4ODD/install /opt/k4ODD/k4ODD
  [[ $? -eq 1 ]] || exit 1          # 1: no match; 0: match; 2: read error
  for lib in /opt/pandora-stack/install/lib/*.so /opt/k4ODD/install/lib/*.so /opt/odd-install/lib/*.so \
             /opt/k4reco-install/lib/*.so; do
    [[ -f "$lib" ]] || exit 1
    if strings "$lib" | grep -E "$pattern"; then
      exit 1
    fi
  done'

echo "== all checks passed"
