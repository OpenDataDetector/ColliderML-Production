#!/usr/bin/env bash
# =============================================================================
# Run the full ColliderML chain for one channel in containers
# =============================================================================
# Usage:
#   ./scripts/cli/run_pipeline_docker.sh --channel ttbar [options]
#
# Chain:
#   ttbar         madgraph_init -> madgraph_generation -> pythia_generation ->
#   higgs_portal  pythia_generation ->
#   both          simulation (Geant4) -> digitization (ACTS tracking, vertexing) ->
#                 particles_full -> pandora_reco -> reco_tables -> package_parquet
#
# Options:
#   --channel NAME        ttbar or higgs_portal (required)
#   --events N            events to produce (default: the configs' value, 10)
#   --pileup MU           mean pile-up interactions per event (default: the pythia
#                         config's value: 0 for ttbar, 10 for higgs_portal)
#   --threads N           threads for Geant4 and ACTS (default: 1)
#   --pandora-processes N concurrent Pandora processes (default: 1). One process needs
#                         about 2 GB without pile-up, about 11 GB per event at pile-up 200
#   --seed N              random seed (default: 42)
#   --output DIR          output directory (default: ./output)
#   --cache DIR           cache directory (default: <repo>/.cache)
#   --sim-image IMAGE     sim image (default: scripts/cli/docker_images.env)
#   --reco-image IMAGE    reco image (default: scripts/cli/docker_images.env)
#   --runtime NAME        docker, podman or podman-hpc (default: the first one found)
#   --from STAGE          start at this stage (earlier outputs must exist)
#   --to STAGE            stop after this stage
#   --list-stages         print the stages of the channel and exit
#
# Outputs:
#   <output>/parquet/{truth,reco}/<table>/*.parquet   final tables
#   <output>/runs/0/                                  intermediate files of every stage
#   <output>/pipeline_timing.csv                      wall time and peak memory per stage
#
# The environment variables of run_docker.sh (OUTPUT_DIR, CACHE_DIR, SEED, SIM_IMAGE,
# RECO_IMAGE, CONTAINER_RUNTIME) set the same defaults; EVENTS is read as --events.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

CHANNEL=""
EVENTS="${EVENTS:-}"
PILEUP=""
THREADS=1
PANDORA_PROCESSES=1
FROM_STAGE=""
TO_STAGE=""
LIST_STAGES=0

need_value() {
    if [ $# -lt 2 ]; then
        echo "ERROR: $1 needs a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --channel)           need_value "$@"; CHANNEL="$2"; shift 2 ;;
        --events)            need_value "$@"; EVENTS="$2"; shift 2 ;;
        --pileup)            need_value "$@"; PILEUP="$2"; shift 2 ;;
        --threads)           need_value "$@"; THREADS="$2"; shift 2 ;;
        --pandora-processes) need_value "$@"; PANDORA_PROCESSES="$2"; shift 2 ;;
        --seed)              need_value "$@"; export SEED="$2"; shift 2 ;;
        --output)            need_value "$@"; export OUTPUT_DIR="$2"; shift 2 ;;
        --cache)             need_value "$@"; export CACHE_DIR="$2"; shift 2 ;;
        --sim-image)         need_value "$@"; export SIM_IMAGE="$2"; shift 2 ;;
        --reco-image)        need_value "$@"; export RECO_IMAGE="$2"; shift 2 ;;
        --runtime)           need_value "$@"; export CONTAINER_RUNTIME="$2"; shift 2 ;;
        --from)              need_value "$@"; FROM_STAGE="$2"; shift 2 ;;
        --to)                need_value "$@"; TO_STAGE="$2"; shift 2 ;;
        --list-stages)       LIST_STAGES=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ -z "$CHANNEL" ]; then
    usage >&2
    exit 1
fi

for name in EVENTS PILEUP THREADS PANDORA_PROCESSES; do
    value="${!name}"
    if [ -n "$value" ] && ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name,,} must be a non-negative integer, got '$value'" >&2
        exit 1
    fi
done

CONFIG_BASE="configs_development/docker_test/$CHANNEL"
if [ ! -d "$REPO_ROOT/$CONFIG_BASE" ]; then
    echo "ERROR: unknown channel '$CHANNEL'. Available:" >&2
    ls "$REPO_ROOT/configs_development/docker_test/" >&2
    exit 1
fi

# stage | script (relative to scripts/) | config file in $CONFIG_BASE
COMMON_STAGES="
simulation|simulation/ddsim_run.py|simulation_config.yaml
digitization|simulation/digi_and_reco.py|digitization_config.yaml
particles_full|postprocessing/convert_particles_full.py|particles_full_config.yaml
pandora_reco|simulation/pandora_reco.py|pandora_reco_config.yaml
reco_tables|postprocessing/convert_reco_tables.py|reco_tables_config.yaml
package_parquet|postprocessing/package_native_parquet.py|package_parquet_config.yaml"

case "$CHANNEL" in
    ttbar)
        STAGES="madgraph_init|simulation/madgraph_init.py|madgraph_init_config.yaml
madgraph_generation|simulation/madgraph_gen.py|madgraph_generation_config.yaml
pythia_generation|simulation/pythia_gen.py|pythia_config.yaml$COMMON_STAGES" ;;
    higgs_portal)
        STAGES="pythia_generation|simulation/pythia_gen.py|pythia_config.yaml$COMMON_STAGES" ;;
    *)
        echo "ERROR: no stage list for channel '$CHANNEL'" >&2
        exit 1 ;;
esac

STAGE_NAMES="$(echo "$STAGES" | cut -d'|' -f1)"
if [ "$LIST_STAGES" = 1 ]; then
    echo "$STAGE_NAMES"
    exit 0
fi
for name in "$FROM_STAGE" "$TO_STAGE"; do
    if [ -n "$name" ] && ! echo "$STAGE_NAMES" | grep -qx "$name"; then
        echo "ERROR: '$name' is not a stage of channel $CHANNEL. Stages:" >&2
        echo "$STAGE_NAMES" >&2
        exit 1
    fi
done

export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"
export RUN_ID=0
mkdir -p "$OUTPUT_DIR"

# Arguments each stage script gets on top of --config/--output/--output-subdir/--seed.
stage_args() {
    local events_arg=()
    [ -n "$EVENTS" ] && events_arg=(--events "$EVENTS")
    case "$1" in
        madgraph_generation)
            echo "${events_arg[@]}" ;;
        pythia_generation)
            echo "${events_arg[@]}" ${PILEUP:+--pileup "$PILEUP"} ;;
        simulation|digitization)
            echo "${events_arg[@]}" --threads "$THREADS" ;;
        pandora_reco)
            echo --processes "$PANDORA_PROCESSES" ;;
        package_parquet)
            # one run holds all events, and they go into one file per table
            [ -n "$EVENTS" ] && echo --run-size "$EVENTS" --chunk-size "$EVENTS" ;;
    esac
    return 0
}

echo "============================================================"
echo "ColliderML pipeline: $CHANNEL"
echo "  events: ${EVENTS:-config default}   pile-up: ${PILEUP:-config default}   threads: $THREADS"
echo "  output: $OUTPUT_DIR"
echo "============================================================"

started=0
[ -z "$FROM_STAGE" ] && started=1
pipeline_start=$(date +%s)

while IFS='|' read -r stage script config; do
    [ -z "$stage" ] && continue
    [ "$stage" = "$FROM_STAGE" ] && started=1
    if [ "$started" = 1 ]; then
        # shellcheck disable=SC2046
        STAGE_NAME="$stage" "$SCRIPT_DIR/run_docker.sh" "$script" "$CONFIG_BASE/$config" $(stage_args "$stage") </dev/null

        if [ "$stage" = madgraph_generation ]; then
            # madgraph_gen.py stages its unsplit output in runs/all/<run>/; the Pythia
            # merge reads the hard scatter from the run directory.
            mkdir -p "$OUTPUT_DIR/runs/$RUN_ID"
            cp "$OUTPUT_DIR/runs/all/$RUN_ID/events.hepmc.gz" "$OUTPUT_DIR/runs/$RUN_ID/events.hepmc.gz"
        fi
    fi
    [ "$stage" = "$TO_STAGE" ] && break
done <<< "$STAGES"

echo ""
echo "============================================================"
echo "Pipeline finished for $CHANNEL in $(( $(date +%s) - pipeline_start )) s"
echo "  tables:  $OUTPUT_DIR/parquet"
echo "  timing:  $OUTPUT_DIR/pipeline_timing.csv"
echo "  disk:    $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1) in the output directory"
echo "============================================================"
