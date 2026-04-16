#!/usr/bin/env bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

# Purpose: download UniRef50 FASTA with aria2c in hhsuite conda env.
# Args: optional output directory path (string); defaults to data/uniref50.
# Returns: exit code 0 on successful download and gzip integrity check.
OUT_DIR="${1:-data/uniref50}"
URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
OUT_FILE="uniref50.fasta.gz"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "Download URL: ${URL}"
echo "Output path: $(pwd)/${OUT_FILE}"

# Use aria2c from conda env hhsuite.
conda run -n hhsuite aria2c \
  --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=10M \
  --auto-file-renaming=false \
  --file-allocation=none \
  --out="${OUT_FILE}" \
  "${URL}"

echo "Verifying gzip archive integrity..."
gzip -t "${OUT_FILE}"
ls -lh "${OUT_FILE}"
echo "Done."
