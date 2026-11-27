#!/bin/bash
# The base image pairs k4FWCore 1.3 with Gaudi 40. k4FWCore's ApplicationMgr constructs
# EventLoopMgr(Warnings=False), a property Gaudi 40 no longer has, so k4run fails before
# the first algorithm. The patch makes that call optional. The self-test at the end of the
# Containerfile runs k4run and so checks the patched file.
source /opt/colliderml-reco/steps/common.sh
load_key4hep

k4fwcore_prefix="$(python3 -c 'import k4FWCore, pathlib; print(pathlib.Path(k4FWCore.__file__).parents[2])')"
verify_patch k4fwcore_gaudi40_eventloopmgr.patch
patch -d "$k4fwcore_prefix" -p1 < "$RECO_HOME/patches/k4fwcore_gaudi40_eventloopmgr.patch"
python3 -c 'import k4FWCore.ApplicationMgr'
