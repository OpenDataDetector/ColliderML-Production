#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J beamspot-train
#SBATCH -o /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_%j.out
#SBATCH -e /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_%j.err

# Usage: sbatch train_gpu.sh [PARQUET_BASE] [EXTRA_ARGS...]
# Example: sbatch train_gpu.sh /path/to/parquet --epochs 50 --wandb-name baseline

PARQUET_BASE=${1:-/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet}
shift  # Remove first arg, rest goes to train.py

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
OUTPUT_DIR=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/checkpoints/${SLURM_JOB_ID}

mkdir -p $(dirname $OUTPUT_DIR)
mkdir -p /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

cd ${REPO}/ml/beamspot_studies/training

python train.py \
    --parquet-base ${PARQUET_BASE} \
    --output-dir ${OUTPUT_DIR} \
    --wandb-project colliderml-beamspot \
    "$@"
