#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J beamspot-train
#SBATCH -o /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_%j.out
#SBATCH -e /global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/logs/train_%j.err

# Train track regression transformer on nominal + shifted beam spot datasets.
# Chains 3 runs sequentially: nominal, shifted 300um, shifted 25um.

set -euo pipefail

REPO=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev
TRAIN_SCRIPT=${REPO}/ml/beamspot_studies/training/train.py
SIM_BASE=/global/cfs/cdirs/m4958/data/ColliderML/simulation
OUT_BASE=/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p ${OUT_BASE}/logs

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate collider-env

echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Job $SLURM_JOB_ID started at $(date) ==="

COMMON_ARGS="--batch-size 256 --lr 1e-3 --epochs 50 --num-workers 4 --patience 10"

# --- Run 1: Nominal (0,0,0) --- ~1M tracks from 16 files
echo ""
echo "=========================================="
echo "Training on NOMINAL dataset"
echo "=========================================="
python ${TRAIN_SCRIPT} \
  --parquet-base ${SIM_BASE}/hard_scatter/ttbar/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_nominal \
  --wandb-name baseline-nominal \
  --max-files 16 \
  ${COMMON_ARGS}

# --- Run 2: Shifted 300um --- ~600K tracks from 50 files
echo ""
echo "=========================================="
echo "Training on SHIFTED 300um dataset"
echo "=========================================="
python ${TRAIN_SCRIPT} \
  --parquet-base ${SIM_BASE}/beamspot_studies/ttbar_shifted_300um/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_shifted_300um \
  --wandb-name baseline-shifted-300um \
  --max-files 50 \
  ${COMMON_ARGS}

# --- Run 3: Shifted 25um --- ~600K tracks from 50 files
echo ""
echo "=========================================="
echo "Training on SHIFTED 25um dataset"
echo "=========================================="
python ${TRAIN_SCRIPT} \
  --parquet-base ${SIM_BASE}/beamspot_studies/ttbar_shifted_25um/v1/parquet \
  --output-dir ${OUT_BASE}/baseline_shifted_25um \
  --wandb-name baseline-shifted-25um \
  --max-files 50 \
  ${COMMON_ARGS}

echo ""
echo "=== All training complete at $(date) ==="
