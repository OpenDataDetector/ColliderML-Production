# Shared helpers for the build steps. Sourced, not executed.
set -euo pipefail

RECO_HOME=/opt/colliderml-reco
JOBS="${BUILD_JOBS:-$(nproc)}"

set -a
source "$RECO_HOME/sources.env"
set +a

# fetch_commit <repo-url> <full-commit-hash> <destination>
# Fetches exactly one commit and verifies the checked-out hash.
fetch_commit() {
  local repo="$1" commit="$2" dest="$3"
  git init --quiet "$dest"
  git -C "$dest" remote add origin "$repo"
  git -C "$dest" fetch --quiet --depth 1 origin "$commit"
  git -C "$dest" checkout --quiet --detach FETCH_HEAD
  local head
  head="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$head" != "$commit" ]]; then
    echo "fetch_commit: $repo gave $head, expected $commit" >&2
    return 1
  fi
  echo "fetched $repo @ $commit"
}

# verify_patch <patch-name>
# Checks a file in $RECO_HOME/patches against SHA256SUMS.
verify_patch() {
  local name="$1"
  (cd "$RECO_HOME/patches" && grep " $name\$" SHA256SUMS | sha256sum --check --quiet --strict)
}

# apply_patch <source-dir> <patch-name>
apply_patch() {
  local dir="$1" name="$2"
  verify_patch "$name"
  git -C "$dir" apply --whitespace=nowarn "$RECO_HOME/patches/$name"
  echo "applied $name"
}

# The spack-generated environment scripts read unset variables.
load_key4hep() {
  set +u
  source /opt/key4hep-shim.sh
  set -u
}
