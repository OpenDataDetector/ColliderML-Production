#!/bin/bash

#SBATCH -A m3443 -q debug
#SBATCH -C cpu
#SBATCH -t 20:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH -o logs/%x-%j.out
#SBATCH -J acts_generate
#SBATCH --module=cvmfs

cd $HOME
# Run everything in a single srun command to ensure proper container environment
srun shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 --module=cvmfs bash -c "
cd /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \
export ATLAS_LOCAL_ROOT_BASE=\$PWD && \
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh && \
lsetup \"views LCG_106 x86_64-el9-gcc13-opt\" && \
cd /global/cfs/cdirs/m3443/usr/dtmurnane/Side_Work/ACTS && \
source build/python/setup.sh && \
python acts/Examples/Scripts/Python/full_chain_odd_anyprocess.py --config $1
"