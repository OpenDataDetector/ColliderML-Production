#!/bin/bash
# Write the build record printed by "colliderml-reco info".
source /opt/colliderml-reco/steps/common.sh

{
  echo "version: ${RECO_VERSION}"
  echo "base_image: ${BASE_IMAGE_NAME}@${BASE_IMAGE_DIGEST}"
  echo "pandora_variant: ${PANDORA_VARIANT}"
  if [[ "$PANDORA_VARIANT" == optimised ]]; then
    echo "pandora_cxx_flags: ${PANDORA_CXX_FLAGS}"
    echo "pandora_ld_flags: ${PANDORA_LD_FLAGS}"
  fi
  echo "sources:"
  grep -E '_(REPO|COMMIT)=' "$RECO_HOME/sources.env" | sed -E 's/[[:space:]]+#.*//; s/^/  /'
  echo "libraries:"
  sha256sum \
    /opt/pandora-stack/install/lib/libPandoraSDK.so \
    /opt/pandora-stack/install/lib/libLCContent.so \
    /opt/pandora-stack/install/lib/libk4GaudiPandoraPlugins.so \
    /opt/pandora-stack/install/lib/libk4DetectorPerformancePFlowPlugins.so \
    /opt/k4ODD/install/lib/libk4ODDPlugins.so \
    /opt/odd-install/lib/libOpenDataDetector.so \
    /opt/k4reco-install/lib/libGaudiTrkUtils.so | sed 's/^/  /'
  echo "settings:"
  sha256sum \
    /opt/k4ODD/k4ODD/options/ODDreconstruction.py \
    /opt/k4ODD/k4ODD/options/PandoraSettingsCLD.xml \
    /opt/k4ODD/k4ODD/options/PandoraLikelihoodData9EBin_zuds200.xml \
    /opt/odd-install/share/OpenDataDetector/xml/OpenDataDetector.xml | sed 's/^/  /'
} > "$RECO_HOME/MANIFEST"
cat "$RECO_HOME/MANIFEST"
