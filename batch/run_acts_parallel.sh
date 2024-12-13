#!/bin/bash

#SBATCH -A atlas -q regular
#SBATCH -C cpu
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -c 8
#SBATCH -o logs/%x-%j.out
#SBATCH -J acts_generate
#SBATCH --module=cvmfs

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch run_acts.sh /path/to/config.yaml"
    exit 1
fi

CONFIG_FILE=$1

# Print config contents
echo "Contents of config file $CONFIG_FILE:"
cat $CONFIG_FILE
echo "---"

# Run everything in a single srun command to ensure proper container environment
cd $HOME
export SLURM_CPU_BIND="cores"
srun --exact -u shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9 --module=cvmfs bash -c "
cd /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && \
export ATLAS_LOCAL_ROOT_BASE=\$PWD && \
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh && \
cd /global/cfs/cdirs/m3443/usr/dtmurnane/Side_Work/ACTS && \
source acts/CI/setup_cvmfs_lcg.sh && \
source build/python/setup.sh && \
python acts/Examples/Scripts/Python/full_chain_odd_anyprocess.py \
    --config $CONFIG_FILE \
    --output-subdir proc_\${SLURM_PROCID} \
    --seed \$((SLURM_PROCID + 1))
"
