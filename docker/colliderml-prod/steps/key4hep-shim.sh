# key4hep setup shim (build-time AND runtime).
# Makes the in-image Gaudi + k4FWCore + spack edm4hep/podio/dd4hep/root look like a
# key4hep environment, so build_pandora_stack.sh (sources $KEY4HEP_SETUP) and the
# calo/pandora pipeline stages resolve k4run / Gaudi / podio / dd4hep off it.
source /opt/build-env.sh
# Gaudi + k4FWCore + the legacy tracking stack (LCIO/KalTest/DDKalTest/k4Reco) that
# k4GaudiPandora's DDTrackCreatorBase needs via k4Reco::GaudiTrkUtils.
export CMAKE_PREFIX_PATH="/opt/gaudi-install:/opt/k4fwcore-install:/opt/ilcutil-install:/opt/lcio-install:/opt/kaltest-install:/opt/ddkaltest-install:/opt/k4reco-install:${CMAKE_PREFIX_PATH:-}"
_trklibs="/opt/ilcutil-install/lib:/opt/ilcutil-install/lib64:/opt/lcio-install/lib:/opt/lcio-install/lib64:/opt/kaltest-install/lib:/opt/kaltest-install/lib64:/opt/ddkaltest-install/lib:/opt/ddkaltest-install/lib64:/opt/k4reco-install/lib:/opt/k4reco-install/lib64"
# Spack runtime libs on LD_LIBRARY_PATH: needed at build time so Pandora/k4ODD
# genconf can dlopen plugins (boost/root/python/tbb); harmless at runtime where
# setup_container_env.sh already sets them.
_spacklibs="$(ls -d /spack/opt/spack/linux-x86_64/*/lib /spack/opt/spack/linux-x86_64/*/lib64 /spack/opt/spack/linux-x86_64/*/lib/root 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="/opt/gaudi-install/lib:/opt/gaudi-install/lib64:/opt/k4fwcore-install/lib:/opt/k4fwcore-install/lib64:${_trklibs}:${_spacklibs}${LD_LIBRARY_PATH:-}"
export PATH="/opt/gaudi-install/bin:/opt/k4fwcore-install/bin:$PATH"
_pyv=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin 2>/dev/null | head -1)
[ -n "$_pyv" ] && export PATH="$_pyv:$PATH"
export PYTHONPATH="/opt/gaudi-install/python:/opt/k4fwcore-install/python:${PYTHONPATH:-}"
# ROOTSYS + module-path fix so genconf (build) and k4run (runtime) cling init
# finds ROOT's C++ system modules (buildcache ROOT bakes a stale build path).
source /opt/steps/root_fix.sh
