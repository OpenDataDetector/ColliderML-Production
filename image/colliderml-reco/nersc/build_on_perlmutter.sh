#!/bin/bash
# Build the image on a Perlmutter compute node. With SAVE_TARBALL=1, also squash it into the
# shared podman-hpc store and export it to TARBALL_DIR as colliderml_reco_VERSION.tar
# (docker-archive) and colliderml_reco_VERSION.oci.tar (OCI archive, used for the push).
# Run inside an allocation:
#   srun -N1 -n1 nersc/build_on_perlmutter.sh VERSION TARBALL_DIR [BUILD_ARGS...]
# BUILD_ARGS go to build.sh, for example --build-arg PANDORA_VARIANT=reference.
#
# podman-hpc cannot read a build context from CFS on compute nodes, so the context is
# copied to node-local /tmp first.
set -euo pipefail

version="${1:?usage: build_on_perlmutter.sh VERSION TARBALL_DIR [BUILD_ARGS...]}"
tarball_dir="${2:?usage: build_on_perlmutter.sh VERSION TARBALL_DIR [BUILD_ARGS...]}"
recipe="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="localhost/colliderml/reco:${version}"
tarball="${tarball_dir}/colliderml_reco_${version}.tar"

context="$(mktemp -d /tmp/reco-image-context.XXXXXX)"
cp -r "$recipe"/{.dockerignore,Containerfile,build.sh,LICENSE,LICENSES,THIRD_PARTY.md,sources.env,requirements.txt,bin,env,patches,steps} "$context/"
chmod -R u+rwX,go+rX "$context"

# Plain podman, so the image carries no podman-hpc settings. --layers=false commits once at
# the end: on these nodes every intermediate layer commit takes about two minutes.
CONTAINER_ENGINE="$recipe/nersc/podman-plain" IMAGE_NAME=colliderml/reco \
  "$context/build.sh" "$version" --layers=false "${@:3}"

podman-hpc image inspect --format '{{.Id}}' "$image"

if [[ "${SAVE_TARBALL:-0}" == 1 ]]; then
  oci="${tarball%.tar}.oci.tar"
  for file in "$tarball" "$oci"; do
    [[ -e "$file" ]] && { echo "refusing to overwrite $file" >&2; exit 1; }
  done
  # Squash the image into the shared store so other nodes can run it.
  podman-hpc migrate "$image"
  # docker-archive for "podman load"; OCI archive with compressed layers for the registry push.
  "$recipe/nersc/podman-plain" save --format docker-archive --output "$tarball" "$image"
  "$recipe/nersc/podman-plain" save --format oci-archive --output "$oci" "$image"
  (cd "$tarball_dir" && for file in "$tarball" "$oci"; do
     sha256sum "$(basename "$file")" > "$(basename "$file").sha256"
   done)
  skopeo inspect --raw "oci-archive:$oci" > "$context/manifest.json"
  test -s "$context/manifest.json"
  echo "OCI manifest digest: sha256:$(sha256sum < "$context/manifest.json" | cut -d' ' -f1)"
  ls -l "$tarball" "$oci"
fi
