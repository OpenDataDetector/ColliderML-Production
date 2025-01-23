#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Array of core counts to test
CORE_COUNTS=(2 4 8 16 32 64 128)

# Submit a job for each core count
for cores in "${CORE_COUNTS[@]}"; do
    config_file="/global/cfs/cdirs/m3443/usr/dtmurnane/Side_Work/ACTS/colliderml_dev/configs_development/testing_and_validation/parallel_tests/snakemake_test_${cores}cpu.yaml"
    echo "Submitting job with $cores cores using config: $config_file"
    sbatch --ntasks-per-node=$cores \
           --cpus-per-task=$((128/$cores)) \
           run_snakemake_generic.sh $config_file
done