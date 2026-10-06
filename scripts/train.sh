#!/bin/bash
set -e

# ============================================================
# Training job script for HKUST COMPUTE
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Domain-Reasoner SFT Job ==="
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate conda
source "$PROJECT_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate reasoner

# Optional: show GPU info
nvidia-smi || true

# Run training
python scripts/train_sft.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo "=== Training finished ==="