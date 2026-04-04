#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J beamspot-train-v2
#SBATCH -o /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_v2_%j.out
#SBATCH -e /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_v2_%j.err

set -euo pipefail

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
SIM_BASE=/global/cfs/cdirs/m4958/data/ColliderML/simulation
OUT_BASE=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

cd ${REPO}/ml/beamspot_studies/training

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Job $SLURM_JOB_ID started at $(date) ==="

COMMON="--batch-size 256 --lr 1e-3 --epochs 50 --num-workers 0 --patience 10"

echo "=== Nominal (resume from epoch 17) ==="
python train.py --parquet-base ${SIM_BASE}/hard_scatter/ttbar/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_nominal_v2 \
  --wandb-name baseline-nominal-v2 --max-files 16 ${COMMON}

echo "=== Shifted 300um ==="
python train.py --parquet-base ${SIM_BASE}/beamspot_studies/ttbar_shifted_300um/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_shifted_300um_v2 \
  --wandb-name baseline-shifted-300um-v2 --max-files 50 ${COMMON}

echo "=== Shifted 25um ==="
python train.py --parquet-base ${SIM_BASE}/beamspot_studies/ttbar_shifted_25um/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_shifted_25um_v2 \
  --wandb-name baseline-shifted-25um-v2 --max-files 50 ${COMMON}

echo "=== All done at $(date) ==="
