#!/bin/bash
# k4FWCore from source (needs Gaudi + edm4hep + podio, all now present).
set -e
source /opt/build-env.sh
export CMAKE_PREFIX_PATH="/opt/gaudi-install:$CMAKE_PREFIX_PATH"
# genconf dlopens built plugins -> spack runtime libs must be on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="/opt/gaudi-install/lib:/opt/gaudi-install/lib64:$(ls -d /spack/opt/spack/linux-x86_64/*/lib /spack/opt/spack/linux-x86_64/*/lib64 /spack/opt/spack/linux-x86_64/*/lib/root 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
git clone --depth 1 https://github.com/key4hep/k4FWCore.git /opt/k4fwcore-src
cmake -S /opt/k4fwcore-src -B /tmp/k4fw-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/k4fwcore-install -DBUILD_TESTING=OFF
cmake --build /tmp/k4fw-build -j4 --target install
rm -rf /tmp/k4fw-build
echo "k4FWCore built"
