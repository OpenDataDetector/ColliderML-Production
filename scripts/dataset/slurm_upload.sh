#!/bin/bash
# Upload ColliderML dataset to HuggingFace. Set HUGGINGFACE_TOKEN before submitting.
# Replace hard-scatter ttbar (100k -> 1M): pass --replace-configs and the four ttbar_pu0_* config names.

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

# Pass any extra args through (e.g. --replace-configs ttbar_pu0_particles ttbar_pu0_tracks ...)
# Example: sbatch slurm_upload.sh --replace-configs ttbar_pu0_particles ttbar_pu0_tracker_hits ttbar_pu0_calo_hits ttbar_pu0_tracks
python upload_to_hf_unified.py unified_dataset_config.yaml "$@"