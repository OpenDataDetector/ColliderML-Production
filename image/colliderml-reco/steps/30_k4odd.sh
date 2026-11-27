#!/bin/bash
# k4ODD: the Gaudi options (digitisation, Pandora settings, calibration constants, photon
# likelihood data) and the k4ODD plugins. The options are read from the source tree at
# /opt/k4ODD/k4ODD/options.
#
# The patch moves the upper DDCaloDigi layer edges to 49 (ECAL) and 37 (HCAL). ODD numbers
# layers from 1, so with edges 48 and 36 the last layer of each calorimeter got calibration 0.
# The second patch sets ECALGapCorrection = 0: ODD has no dead gaps between ECAL cells, the
# correction never changed a hit in our validation, and its code reads the hits through
# expired handles (key4hep/k4GaudiPandora#68).
source /opt/colliderml-reco/steps/common.sh
load_key4hep
set +u; source /opt/pandora-stack/install/setup_stack.sh; set -u

fetch_commit "$K4ODD_REPO" "$K4ODD_COMMIT" /opt/k4ODD
apply_patch /opt/k4ODD k4odd_calo_last_layer.patch
apply_patch /opt/k4ODD k4odd_ecal_gap_correction_off.patch

cmake -S /opt/k4ODD -B /tmp/k4odd-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4ODD/install
cmake --build /tmp/k4odd-build -j"$JOBS" --target install
rm -rf /tmp/k4odd-build

test -f /opt/k4ODD/install/lib/libk4ODDPlugins.so
test -f /opt/k4ODD/k4ODD/options/PandoraLikelihoodData9EBin_zuds200.xml
