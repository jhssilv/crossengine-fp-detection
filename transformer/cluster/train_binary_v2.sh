#!/bin/bash
#SBATCH --job-name=fp_transformer_v2
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/jhdssilva/tcc/transformer/logs/train_v2_%j.log
#SBATCH --error=/home/users/jhdssilva/tcc/transformer/logs/train_v2_%j.err

set -euo pipefail

CLUSTER_ROOT="$HOME/tcc/transformer"
PYTHON="$HOME/tcc/miniconda/envs/transformer/bin/python"
SCRIPTS="$CLUSTER_ROOT/scripts"
DATA="$CLUSTER_ROOT/data/translated_dataset.parquet"
EMBEDDINGS="$CLUSTER_ROOT/data/fasttext.vec"
OUT="$CLUSTER_ROOT/model_v2_${SLURM_JOB_ID}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$OUT" "$CLUSTER_ROOT/logs"

log "Job ID: ${SLURM_JOB_ID}"
log "Node:   $(hostname)"
log "GPUs:   ${CUDA_VISIBLE_DEVICES:-unset}"
log "Python: $("$PYTHON" --version)"

# ── GPU sanity check ──────────────────────────────────────────────────────
log "GPU check:"
"$PYTHON" -c "
import torch
print(f'  torch {torch.__version__}  CUDA built={torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM:   {vram:.1f} GB')
"

log "Dataset: $(du -sh "$DATA" | cut -f1)"
log "Output:  ${OUT}"

# ── Train ─────────────────────────────────────────────────────────────────
# train_v2.py is binary-only; threshold tuning is built in.
# Larger settings than the CPU-optimized defaults — GPU can handle them.
log "Starting training (train_v2.py)..."
"$PYTHON" "$SCRIPTS/train_v2.py" \
    --dataset    "$DATA"         \
    --embeddings "$EMBEDDINGS"   \
    --out        "$OUT"          \
    --neg-ratio  10              \
    --max-len    256             \
    --epochs     32              \
    --batch-size 512             \
    --lr         1e-3            \
    --patience   5               \
    --seed       42

log "Training complete."
log "Artifacts:"
ls -lh "$OUT"

# ── Package for download ──────────────────────────────────────────────────
log "Packaging artifacts..."
TARBALL="$CLUSTER_ROOT/transformer_v2_${SLURM_JOB_ID}.tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "Tarball: $(du -sh "$TARBALL" | cut -f1) → ${TARBALL}"
log "Done."
