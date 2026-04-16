# Personal Training Notes (LBSTER)

This note summarizes what `slurm/personal/train_env_check.sh` actually runs, which training configs are in effect, and what to use for from-scratch practice.

## 1) What `train_env_check.sh` currently does

`slurm/personal/train_env_check.sh` launches:

```bash
uv run lobster_train \
  data.path_to_fasta="test_data/query.fasta" \
  logger=wandb \
  logger.project="${WANDB_PROJECT}" \
  logger.name="${WANDB_NAME}" \
  paths.root_dir="."
```

Important: it does not override epochs, lr, warmup, scheduler, model size, masking, or batch size. Those come from Hydra defaults.

## 2) Effective Training Config (for this script)

From `src/lobster/hydra_config/train.yaml` defaults:
- `data: fasta.yaml`
- `model: mlm.yaml`
- `trainer: default.yaml`
- `callbacks: default.yaml`
- `logger: wandb.yaml`
- `setup: default.yaml`
- `paths: default.yaml`

### Trainer
From `src/lobster/hydra_config/trainer/default.yaml`:
- `accelerator: gpu`
- `devices: 1`
- `precision: 32`
- `accumulate_grad_batches: 1`
- `max_steps: 100_000_000`
- `val_check_interval: 0.25`
- `limit_val_batches: 10`
- `num_sanity_val_steps: 2`

Notes:
- This is step-based training (no explicit epoch cap set in config).
- Effective stop condition is `max_steps` unless interrupted.

### Model (default MLM path)
From `src/lobster/hydra_config/model/mlm.yaml`:
- `_target_: lobster.model.LobsterPMLM`
- `lr: 1e-3`
- `model_name: esm2_t6_8M_UR50D`
- `mask_percentage: 0.25`
- `num_warmup_steps: 10_000`
- `num_training_steps: ${trainer.max_steps}`
- `max_length: 512`
- `tokenizer_dir: pmlm_tokenizer`

From `src/lobster/model/_mlm.py`:
- optimizer: `AdamW`
- optimizer defaults used here: `beta1=0.9`, `beta2=0.98`, `eps=1e-12`
- scheduler default in code path: `constant_with_warmup` (step interval)
- masking: random token mask each step, excluding `CLS/PAD/EOS`

### Data
From `src/lobster/hydra_config/data/fasta.yaml`:
- `_target_: lobster.data.FastaLightningDataModule`
- `batch_size: 64`
- `num_workers: 1`
- `max_length: ${model.max_length}`
- `mlm: True`

From `src/lobster/data/_fasta_datamodule.py`:
- default split if one FASTA file: train/val/test = `0.90 / 0.05 / 0.05`
- current pre-split detection checks `if any(["train" in self._path_to_fasta])`, which usually does **not** match normal file paths, so in practice you typically get random split behavior
- shuffle enabled for train loader

### Setup
From `src/lobster/hydra_config/setup/seed/default.yaml`:
- seed: `0xf1eece`
- workers: true

From `src/lobster/hydra_config/setup/torch/default.yaml`:
- `torch.set_float32_matmul_precision("medium")`

## 3) Pretrained vs From-Scratch (important)

Default `model_name=esm2_t6_8M_UR50D` is loaded from pretrained Facebook ESM2 weights (`from_pretrained`), so this is not a pure from-scratch run.

For from-scratch MLM in this repo, set `model.model_name` to one of internal configs:
- `MLM_mini`
- `MLM_11M`
- `MLM_24M`
- `MLM_68M`
- `MLM_83M`
- `MLM_113M`
- `MLM_150M`
- `MLM_650M`
- `MLM_3B`

Those are defined in `src/lobster/model/_mlm_configuration.py`.

## 4) Flash-Attn Findings (H100)

- Your current default training path (`model=mlm`) does not expose an explicit flash-attn toggle and uses classic attention code in `lm_base`.
- Explicit flash-attn support in this repo is strongest in:
  - `UME` / `FlexBERT` (`use_flash_attn` / `use_fa2`)
  - `NeoBERT` internals (flash-attn varlen path)

Practical implication:
- If your goal is "MLM from scratch on FASTA quickly", use `model=mlm` (simplest path), but it is not the repo's most explicit flash-attn path.
- If your goal is "practice flash-attn training", prefer UME/FlexBERT/NeoBERT workflows.

## 5) Command Templates for Practice

### A) From-scratch MLM on a small FASTA (simple and stable)

```bash
uv run lobster_train \
  data.path_to_fasta="/abs/path/family_100k.fasta" \
  model.model_name=MLM_11M \
  model.lr=3e-4 \
  model.mask_percentage=0.15 \
  model.num_warmup_steps=2000 \
  trainer.max_steps=50000 \
  data.batch_size=128 \
  data.max_length=512 \
  logger=csv \
  paths.root_dir="."
```

### B) Same idea for larger corpus (e.g., UniRef50 subset)

```bash
uv run lobster_train \
  data.path_to_fasta="/abs/path/uniref50_subset.fasta" \
  model.model_name=MLM_24M \
  model.lr=2e-4 \
  model.mask_percentage=0.15 \
  model.num_warmup_steps=5000 \
  trainer.max_steps=200000 \
  data.batch_size=256 \
  data.max_length=512 \
  logger=wandb \
  logger.project="lobster_pretrain_practice" \
  paths.root_dir="."
```

### C) If you want split control right now

```bash
uv run lobster_train \
  data.path_to_fasta="/abs/path/family_100k.fasta" \
  +data.lengths="[0.95,0.03,0.02]" \
  model.model_name=MLM_11M \
  logger=csv \
  paths.root_dir="."
```

## 6) One subtle config detail

`src/lobster/hydra_config/lr_scheduler/default.yaml` exists, but `lobster_train` currently instantiates scheduler behavior from model args (`model.scheduler`, `model.num_warmup_steps`, etc.). In this default MLM setup, scheduler behavior is controlled by model config/code, not by a separately wired top-level scheduler object.
