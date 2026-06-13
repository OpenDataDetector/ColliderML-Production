# key4hep setup shim (build-time AND runtime).
# Makes the in-image Gaudi + k4FWCore + spack edm4hep/podio/dd4hep/root look like a
# key4hep environment, so build_pandora_stack.sh (sources $KEY4HEP_SETUP) and the
# calo/pandora pipeline stages resolve k4run / Gaudi / podio / dd4hep off it.
source /opt/build-env.sh
export CMAKE_PREFIX_PATH="/opt/gaudi-install:/opt/k4fwcore-install:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="/opt/gaudi-install/lib:/opt/gaudi-install/lib64:/opt/k4fwcore-install/lib:/opt/k4fwcore-install/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/opt/gaudi-install/bin:/opt/k4fwcore-install/bin:$PATH"
_pyv=$(ls -d /spack/opt/spack/linux-x86_64/python-venv-*/bin 2>/dev/null | head -1)
[ -n "$_pyv" ] && export PATH="$_pyv:$PATH"
export PYTHONPATH="/opt/gaudi-install/python:/opt/k4fwcore-install/python:${PYTHONPATH:-}"
