#!/bin/bash
# FALLBACK Pandora driver for the Release-2 pilot: the reco container image tarball
# (stress_mu200/reco_image.tar) no longer exists, so run the June-2026 recipe instead:
# shifter atlas-grid-almalinux9 + cvmfs LCG_107/key4hep + the on-disk k4ODD/k4GaudiPandora
# builds in colliderml-release-2, optimised LCContent preloaded (same basis as the
# 156 ev/node-h mu=200 number). Physics constants = ODDreconstruction.py defaults (final).
#
#   srun -N1 -n1 -c128 shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 \
#        --module=cvmfs /path/to/pandora_shifter.sh
#
# Step A: one process, NEV_SINGLE events -> runs/0/reco_edm4hep.root (input for reco_tables).
# Step B: NPAR concurrent processes x NEV_PAR events each (same events; timing only) -> ev/node-h.
set -o pipefail
RUN=/global/cfs/cdirs/m4958/data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/runs/0
LOGDIR=/global/cfs/cdirs/m4958/data/ColliderML/staging/release2_pilot/ttbar_pu0/v1/logs
NEV_SINGLE=${NEV_SINGLE:-256}
NPAR=${NPAR:-64}
NEV_PAR=${NEV_PAR:-20}
REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-release-2
PSDK=/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/pandorasdk/3.4.2-2j446o/lib/libPandoraSDK.so.03.04
LCC=$REPO/testbed/prebuilt/libLCContent.so
IN=$RUN/sim_with_tracks.root

echo "=== $(date '+%F %T') pandora_shifter start host=$(hostname) cores=$(nproc) memGB=$(free -g | awk '/Mem/{print $2}')"
[ -f "$IN" ] || { echo "missing $IN"; exit 1; }
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc13-opt >/dev/null 2>&1
export KEY4HEP_SETUP=/cvmfs/sw.hsf.org/key4hep/setup.sh
cd "$REPO" && source k4ODD/setup.sh >/dev/null 2>&1 && cd k4ODD || { echo "k4ODD setup failed"; exit 1; }
export LD_PRELOAD="$PSDK $LCC"
export K4ODD_TRACK_CREATOR=DDTrackCreatorCLIC K4ODD_TRACK_COLLECTION=ActsTracks
export K4ODD_PANDORA_SETTINGS="$REPO/k4ODD/k4ODD/options/PandoraSettingsCLD.xml"
export K4ODD_MAX_TRACK_SIGMA_POVERP=999
KR="$(which k4run)"; echo "k4run=$KR python=$(which python)"

# ---- Step A: single process, the file reco_tables will read
t0=$(date +%s)
python "$KR" k4ODD/options/ODDreconstruction.py --inputFile "$IN" \
    --outputFile "$RUN/reco_edm4hep.root" --num-events "$NEV_SINGLE" > "$LOGDIR/pandora_single.log" 2>&1
rc=$?; tA=$(( $(date +%s) - t0 ))
echo "=== $(date '+%F %T') stepA rc=$rc events=$NEV_SINGLE wall=${tA}s per_event=$(python -c "print(round($tA/$NEV_SINGLE,2))")s"
[ $rc -eq 0 ] || { tail -20 "$LOGDIR/pandora_single.log"; exit 2; }
podio-dump "$RUN/reco_edm4hep.root" 2>/dev/null | grep -E "GaudiPandoraPFOs|GaudiPandoraClusters|digiECalBarrel" | head -3

# ---- Step B: NPAR concurrent processes (one wave), throughput in ev/node-h
PT=$RUN/partest_N${NPAR}; mkdir -p "$PT"
t0=$(date +%s)
for i in $(seq 1 $NPAR); do
  ( python "$KR" k4ODD/options/ODDreconstruction.py --inputFile "$IN" \
      --outputFile "$PT/p${i}.root" --num-events "$NEV_PAR" > "$PT/p${i}.log" 2>&1 ) &
done
wait
tB=$(( $(date +%s) - t0 ))
ok=$(grep -l "Application Manager Terminated successfully" "$PT"/p*.log 2>/dev/null | wc -l)
echo "=== $(date '+%F %T') stepB N=$NPAR nev=$NEV_PAR ok=$ok/$NPAR wall=${tB}s ev_per_node_hour=$(python -c "print(round($NPAR*$NEV_PAR*3600/$tB))")"
rm -f "$PT"/p*.root
echo "=== pandora_shifter done"
