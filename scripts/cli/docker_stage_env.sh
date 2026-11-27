#!/usr/bin/env bash
# =============================================================================
# Stage environment for the docker/podman pipeline
# =============================================================================
# Sourced inside the container before a stage script runs:
#
#   source /workspace/scripts/cli/docker_stage_env.sh sim    # sim/tracking image
#   source /workspace/scripts/cli/docker_stage_env.sh reco   # Pandora image
#
# sim  : generation, Geant4, ACTS digitisation and tracking, truth tables, packaging.
#        Uses the OpenDataDetector baked into the image (/opt/odd, /opt/odd-install).
# reco : pandora_reco and reco_tables. Activates the key4hep stack, the Pandora
#        stack and k4ODD that the reco image ships under /opt.
#
# These are the steps of the `simulation` and `pandora_reco` blocks of
# scripts/cli/env_setup.yaml.two-container-template, without any site config.
# =============================================================================

_stage_env_kind="${1:-sim}"
_stage_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_stage_env_fail() {
    echo "ERROR: $1" >&2
    return 1
}

case "$_stage_env_kind" in
    sim)
        if [ ! -f /opt/odd/xml/OpenDataDetector.xml ] || [ ! -f /opt/odd-install/bin/this_odd.sh ]; then
            _stage_env_fail "this image has no OpenDataDetector under /opt/odd and /opt/odd-install; use ghcr.io/opendatadetector/sw:pr-8 or later" || return 1
        fi
        export COLLIDERML_ODD_SOURCE=/opt/odd
        export COLLIDERML_ODD_INSTALL=/opt/odd-install
        # shellcheck disable=SC1091
        source "$_stage_env_dir/setup_container_env.sh" || return 1
        # shellcheck disable=SC1091
        source /opt/odd-install/bin/this_odd.sh
        # Tracking geometry, digitisation and seeding configs come from the same ODD
        # tree as the simulation (digi_and_reco.py reads ODD_GEO_DIR).
        export ODD_GEO_DIR=/opt/odd

        # MadGraph sits under a hashed spack prefix; give the configs a fixed path.
        _mg5_dir=$(ls -d /spack/opt/spack/linux-x86_64/madgraph5amc-* 2>/dev/null | head -1)
        if [ -n "$_mg5_dir" ] && [ ! -e /opt/mg5 ]; then
            ln -s "$_mg5_dir" /opt/mg5
        fi
        unset _mg5_dir

        # scipy is needed by convert_particles_full.py and is not in the image.
        _pip_target="${COLLIDERML_CACHE:-/cache}/pip"
        if ! python3 -c "import scipy" 2>/dev/null; then
            echo "Installing scipy into $_pip_target ..."
            python3 -m pip install --quiet --target="$_pip_target" scipy \
                || _stage_env_fail "could not install scipy (needed for the truth particle tables); check network access" || return 1
            export PYTHONPATH="$_pip_target:${PYTHONPATH:-}"
        fi
        unset _pip_target
        ;;
    reco)
        if [ -f /opt/colliderml-reco/env/setup.sh ]; then
            # shellcheck disable=SC1091
            source /opt/colliderml-reco/env/setup.sh
        else
            # Images built before the setup script existed (localhost/colliderml/reco:20261106).
            for _f in /opt/key4hep-shim.sh /opt/pandora-stack/install/setup_stack.sh /opt/odd-install/bin/this_odd.sh; do
                if [ ! -f "$_f" ]; then
                    _stage_env_fail "$_f not found; the reco stages need the ColliderML reco image" || return 1
                fi
            done
            unset _f
            # shellcheck disable=SC1091
            source /opt/key4hep-shim.sh
            # shellcheck disable=SC1091
            source /opt/pandora-stack/install/setup_stack.sh
            # shellcheck disable=SC1091
            source /opt/odd-install/bin/this_odd.sh
            export K4ODD_PATH=/opt/k4ODD
            export ODD_INSTALL_DIR=/opt/odd-install
            export LD_LIBRARY_PATH="/opt/k4ODD/install/lib:/opt/k4ODD/install/lib64:${LD_LIBRARY_PATH:-}"
            export PYTHONPATH="/opt/k4ODD/install/python:${PYTHONPATH:-}"
        fi
        ;;
    *)
        _stage_env_fail "unknown stage environment '$_stage_env_kind' (expected sim or reco)" || return 1
        ;;
esac

unset _stage_env_kind _stage_env_dir
unset -f _stage_env_fail
