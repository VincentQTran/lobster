#!/usr/bin/env bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

# Usage:
#   sbatch slurm/personal/train_mlm11m_uniref50_cramming.sh /abs/path/to/uniref50.fasta
# or
#   UNIREf_FASTA=/abs/path/to/uniref50.fasta sbatch slurm/personal/train_mlm11m_uniref50_cramming.sh

FASTA_PATH="${1:-${UNIREf_FASTA:-}}"
if [[ -z "${FASTA_PATH}" ]]; then
  echo "ERROR: Provide FASTA path as arg1 or set UNIREf_FASTA"
  exit 2
fi
if [[ ! -f "${FASTA_PATH}" ]]; then
  echo "ERROR: FASTA not found: ${FASTA_PATH}"
  exit 2
fi

export WANDB_PROJECT="lobster_mlm11m_cramming"
export WANDB_NAME="mlm11m_uniref50_${SLURM_JOB_ID:-local}"

RUN_ROOT="slurm/runs/mlm11m_uniref50"
mkdir -p "${RUN_ROOT}"

# Paper-style targets:
# - batch_size=128
# - accumulate_grad_batches=16  => effective batch = 2048 sequences/step
# - max_length=512              => 1,048,576 tokens/step effective
# - max_steps=50,000
# - lr=1e-3, linear schedule, warmup=1,000 steps
# - masking=25%
# - AdamW betas=(0.99, 0.98), eps=1e-12
# - grad clip=0.5
# - mixed precision

echo "python: $(uv run python -c 'import sys; print(sys.executable)')"
echo "fasta:  ${FASTA_PATH}"
echo "wandb:  ${WANDB_PROJECT} / ${WANDB_NAME}"
echo "setup:  batch=128, accum=16, effective_batch=2048, seq_len=512, max_steps=50000"

uv run lobster_train \
  data.path_to_fasta="${FASTA_PATH}" \
  model.model_name=MLM_11M \
  model.lr=1e-3 \
  +model.scheduler=linear \
  model.num_warmup_steps=1000 \
  model.num_training_steps=50000 \
  model.mask_percentage=0.25 \
  +model.beta1=0.99 \
  +model.beta2=0.98 \
  +model.eps=1e-12 \
  model.max_length=512 \
  data.batch_size=128 \
  data.num_workers=8 \
  data.max_length=512 \
  trainer.max_steps=50000 \
  trainer.accumulate_grad_batches=16 \
  trainer.gradient_clip_val=0.5 \
  trainer.precision=bf16-mixed \
  logger=wandb \
  logger.project="${WANDB_PROJECT}" \
  logger.name="${WANDB_NAME}" \
  paths.root_dir="${RUN_ROOT}"
