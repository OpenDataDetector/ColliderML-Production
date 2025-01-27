#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <subdirectory>"
    echo "Example: $0 gg2ttbar"
    exit 1
fi

SUBDIR=$1

# Source directory (use full path)
SRC_DIR="/global/cfs/projectdirs/m3443/usr/dtmurnane/Side_Work/ACTS/outputs/low_pileup_pilot/${SUBDIR}"

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Directory $SRC_DIR does not exist"
    exit 1
fi

# Destination at CERN
DEST="dmurnane@lxplus.cern.ch:/eos/user/d/dmurnane/ColliderML/low_pileup_pilot/${SUBDIR}"

echo "Starting transfer of ${SUBDIR} at $(date)"

# Run the transfer
rsync -avzP \
    --include="*/" \
    --include="*.root" \
    --include="*.csv" \
    --include="*.hepmc3" \
    --exclude="*" \
    ${SRC_DIR}/ ${DEST}/

echo "Transfer completed at $(date)"