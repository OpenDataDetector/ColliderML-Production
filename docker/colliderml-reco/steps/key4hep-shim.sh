# Native key4hep activation for the RECO image (no cvmfs, no from-source shim).
#
# The reco base (ghcr.io/key4hep/key4hep-sim-reco-ubuntu24) ships the whole
# key4hep stack as a GLOBAL spack instance (ROOT+eve, Gaudi, k4FWCore, podio,
# edm4hep, DD4hep, and the prebuilt LCIO/KalTest/DDKalTest/k4Reco our Pandora
# charged-PF chain links). It has NO merged view, so activation = source spack
# + `spack load` the key4hep-stack bundle (emits PATH / LD_LIBRARY_PATH /
# CMAKE_PREFIX_PATH / PYTHONPATH for the full transitive closure).
#
# Lives at /opt/key4hep-shim.sh: the same path the (sw-image) build steps used,
# so build_pandora.sh (KEY4HEP_SETUP) and build_k4odd.sh source it unchanged,
# and the calo_digitization / pandora_reco runtime env blocks reuse it verbatim.
source /opt/setup_spack.sh
eval "$(spack load --sh key4hep-stack)"
# Source-built k4Reco (build_k4reco.sh): the base's spack k4reco-0.2.1 ships the lib
# but NO CMake export, so find_package(k4Reco) / k4Reco::GaudiTrkUtils cannot resolve.
# Put our /opt/k4reco-install FIRST so its real k4RecoConfig.cmake + imported target
# win at build time, and libGaudiTrkUtils.so resolves when k4run dlopens the Pandora
# plugin at reco time.
export CMAKE_PREFIX_PATH="/opt/k4reco-install:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/opt/k4reco-install/lib:/opt/k4reco-install/lib64:${LD_LIBRARY_PATH:-}"
# Merged-include CPATH (cvmfs-view equivalent). The calibrated Pandora stack was
# developed against the key4hep cvmfs VIEW, where every package's headers live in one
# include/ — so k4GaudiPandora does cross-package includes (e.g. <kaltest/TKalDetCradle.h>
# pulled in transitively via k4Reco's public GaudiDDKalTest.h) WITHOUT find_package-ing
# those packages. The spack-load model keeps per-prefix includes, so replicate the merged
# view by putting every loaded prefix's include/ on CPATH for the build. Harmless at
# runtime (no compilation). k4reco-install first so the flat <k4Reco/*.h> symlinks win.
export CPATH="/opt/k4reco-install/include:$(echo "${CMAKE_PREFIX_PATH}" | tr ':' '\n' | sed 's:$:/include:' | tr '\n' ':')${CPATH:-}"
