#!/bin/bash

#SBATCH --account             m4958
#SBATCH --constraint          cpu
#SBATCH --cpus-per-task       256
#SBATCH --error               /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/dataset/logs/job_0_%j.err
#SBATCH --job-name            huggingface_upload
#SBATCH --nodes               1
#SBATCH --ntasks-per-node     1
#SBATCH --output              /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/dataset/logs/job_0_%j.out
#SBATCH --qos                 regular
#SBATCH --time                24:00:00

cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/scripts/dataset
eval "$(conda shell.bash hook)"
conda activate collider-env

python upload_to_hf_unified.py unified_dataset_config.yaml