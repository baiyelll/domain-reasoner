#!/bin/bash
set -e

# ============================================================
# One-time environment setup for HKUST COMPUTE
# Run this once via a job. It installs Miniconda + all packages
# into the project directory.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# 1. Install Miniconda if not present
if [ ! -d "miniconda" ]; then
    echo ">>> Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$PROJECT_ROOT/miniconda"
    rm miniconda.sh
fi

# 2. Source conda
source "$PROJECT_ROOT/miniconda/etc/profile.d/conda.sh"

# 3. Create environment
if ! conda env list | grep -q "reasoner"; then
    echo ">>> Creating conda environment 'reasoner'..."
    conda create -y -n reasoner python=3.11
fi

conda activate reasoner

# 4. Install PyTorch (CUDA 12.1 / 12.4 compatible – adjust if needed)
echo ">>> Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install the rest
echo ">>> Installing project requirements..."
pip install -r requirements.txt

# 6. Extra Unsloth optimizations (optional but recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || true

echo ">>> Environment setup complete!"
echo "To activate later: source $PROJECT_ROOT/miniconda/etc/profile.d/conda.sh && conda activate reasoner"