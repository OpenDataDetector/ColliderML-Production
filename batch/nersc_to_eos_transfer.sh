#!/bin/bash

# Source directory (use full path)
SRC_DIR="/global/cfs/projectdirs/m3443/usr/dtmurnane/Side_Work/ACTS/outputs/low_pileup_pilot"

# Destination at CERN
DEST="dmurnane@lxplus.cern.ch:/eos/user/d/dmurnane/ColliderML/"

# Run the transfer
rsync -avzP \
    --include="*/" \
    --include="*.root" \
    --include="*.csv" \
    --include="*.hepmc3" \
    --exclude="*" \
    ${SRC_DIR} ${DEST}

# Print completion message
echo "Transfer completed at $(date)"