#!/bin/bash
# Run once on the PCAD login node (NOT via SLURM) to install Miniconda and
# all Python dependencies into the shared NFS home.
#
# Why Miniconda:
#   blaise compute nodes have Python 3.7.3 (too old for PyTorch 2.x).
#   The login node has Python 3.11.2 and internet access.
#   A Miniconda install in ~/tcc/miniconda lives on the NFS-mounted /home,
#   so blaise can use its absolute binary path without any system Python.
#
# Usage (from login node, NOT inside a SLURM job):
#   bash ~/tcc/transformer/cluster/setup_env.sh
#
# Creates:
#   ~/tcc/miniconda/            — Miniconda base
#   ~/tcc/miniconda/envs/transformer/  — Python 3.11 + torch (cu118) + deps
#
# Target hardware: blaise — 4× Tesla P100-SXM2 16 GB, CUDA 11.8, x86_64

set -euo pipefail

MINICONDA_DIR="$HOME/tcc/miniconda"
ENV_NAME="transformer"
ENV_PYTHON="$MINICONDA_DIR/envs/$ENV_NAME/bin/python"
PIP="$MINICONDA_DIR/envs/$ENV_NAME/bin/pip"
CONDA="$MINICONDA_DIR/bin/conda"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Install Miniconda (skip if already present) ────────────────────────
if [ -f "$CONDA" ]; then
    log "Miniconda already at ${MINICONDA_DIR} — skipping install."
else
    log "Downloading Miniconda (x86_64)..."
    INSTALLER="/tmp/miniconda_installer.sh"
    curl -fsSL \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -o "$INSTALLER"

    log "Installing Miniconda to ${MINICONDA_DIR}..."
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$INSTALLER"
    log "Miniconda installed."
fi

# ── 2. Create conda env with Python 3.11 (skip if exists) ────────────────
if [ -d "$MINICONDA_DIR/envs/$ENV_NAME" ]; then
    log "Conda env '${ENV_NAME}' already exists — skipping creation."
else
    log "Creating conda env '${ENV_NAME}' with Python 3.11 + pip (conda-forge)..."
    # Use conda-forge to avoid Anaconda's ToS requirement for pkgs/main
    # pip must be listed explicitly — conda-forge doesn't bundle it by default
    "$CONDA" create -n "$ENV_NAME" python=3.11 pip \
        --override-channels -c conda-forge \
        -y --quiet
fi

# Ensure pip is present even if the env was created without it
if [ ! -f "$PIP" ]; then
    log "pip missing from env — installing via conda..."
    "$CONDA" install -n "$ENV_NAME" pip \
        --override-channels -c conda-forge \
        -y --quiet
fi

# ── 3. Install PyTorch for CUDA 11.8 ─────────────────────────────────────
log "Installing PyTorch (CUDA 11.8)..."
"$PIP" install torch \
    --index-url https://download.pytorch.org/whl/cu118 \
    --quiet

log "Verifying torch..."
"$ENV_PYTHON" -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA built with: {torch.version.cuda}')
print(f'  CUDA available (login node, no GPU): {torch.cuda.is_available()}')
"

# ── 4. Install other dependencies ─────────────────────────────────────────
log "Installing pandas, pyarrow, scikit-learn, numpy..."
"$PIP" install \
    "pandas>=2.0" \
    "pyarrow>=14.0" \
    "scikit-learn>=1.3" \
    "numpy>=1.24" \
    --quiet

log "Verifying deps..."
"$ENV_PYTHON" -c "
import pyarrow, sklearn, numpy
print(f'  pyarrow    {pyarrow.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  numpy      {numpy.__version__}')
"

log ""
log "Setup complete."
log "Python binary (use this in SBATCH): ${ENV_PYTHON}"
log "Test on blaise compute node:"
log "  ${ENV_PYTHON} -c \"import torch; print(torch.cuda.is_available())\""
