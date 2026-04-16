#!/usr/bin/env bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

# Purpose: quick end-to-end training smoke test for LBSTER on SLURM GPU.
# Args: none (bash script; uses repo-local paths and Hydra CLI overrides).
# Returns: exit code 0 on success, non-zero on failure.
export WANDB_PROJECT="lobster_test"
export WANDB_NAME="train_env_check_${SLURM_JOB_ID:-local}"

echo "python: $(uv run python -c 'import sys; print(sys.executable)')"
echo "wandb project: ${WANDB_PROJECT}"

uv run lobster_train \
  data.path_to_fasta="test_data/query.fasta" \
  logger=wandb \
  logger.project="${WANDB_PROJECT}" \
  logger.name="${WANDB_NAME}" \
  paths.root_dir="." 