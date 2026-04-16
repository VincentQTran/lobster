#!/usr/bin/env bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

uv run python -c "import sys; print(sys.executable)"
uv run python -c "import flash_attn; print(flash_attn.__version__)"

nvidia-smi

which nvcc
nvcc --version

uv run python -c "import torch; from flash_attn.flash_attn_interface import flash_attn_func; print('cuda:', torch.cuda.is_available()); print('ok: flash_attn_func imported')"