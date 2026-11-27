# key4hep environment of the base image, plus the k4Reco build in /opt/k4reco-install.
# Build steps and the runtime environment (env/setup.sh) both start from this file.
source /opt/colliderml-reco/env/key4hep-stack.sh

export CMAKE_PREFIX_PATH="/opt/k4reco-install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="/opt/k4reco-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# k4GaudiPandora includes headers of packages it does not find_package (for example
# <kaltest/TKalDetCradle.h> through k4Reco), which works in a merged key4hep view but not
# with one prefix per package. Put every prefix's include directory on CPATH.
export CPATH="/opt/k4reco-install/include:$(echo "${CMAKE_PREFIX_PATH}" | tr ':' '\n' | sed 's:$:/include:' | paste -sd: -)${CPATH:+:$CPATH}"
