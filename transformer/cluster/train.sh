#!/bin/bash
#SBATCH --job-name=fp_transformer
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/jhdssilva/tcc/transformer/logs/train_%j.log
#SBATCH --error=/home/users/jhdssilva/tcc/transformer/logs/train_%j.err

set -euo pipefail

CLUSTER_ROOT="$HOME/tcc/transformer"
PYTHON="$HOME/tcc/miniconda/envs/transformer/bin/python"
SCRIPTS="$CLUSTER_ROOT/scripts"
DATA="$CLUSTER_ROOT/data/translated_dataset.parquet"
EMBEDDINGS="$CLUSTER_ROOT/data/fasttext.vec"
OUT="$CLUSTER_ROOT/model_${SLURM_JOB_ID}"

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

# ── Dataset check ─────────────────────────────────────────────────────────
log "Dataset: $(du -sh "$DATA" | cut -f1)"
log "Output:  ${OUT}"

# ── Train ─────────────────────────────────────────────────────────────────
log "Starting training..."
PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/train.py" \
    --dataset        "$DATA" \
    --embeddings     "$EMBEDDINGS" \
    --out            "$OUT" \
    --neg-ratio      10 \
    --max-len        512 \
    --num-layers     2 \
    --epochs         16 \
    --batch-size     128 \
    --lr             3e-4 \
    --warmup-epochs  2 \
    --gamma          2.0 \
    --weight-decay   1e-4 \
    --patience       3 \
    --num-workers    8

log "Training complete."

# ── Threshold tuning ──────────────────────────────────────────────────────
log "Tuning per-label decision thresholds..."
PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/tune_threshold.py" \
    --model-dir    "$OUT" \
    --dataset      "$DATA" \
    --embeddings   "$EMBEDDINGS" \
    --num-workers  8

log "Artifacts:"
ls -lh "$OUT"

# ── Package for download ──────────────────────────────────────────────────
log "Packaging artifacts..."
TARBALL="$CLUSTER_ROOT/transformer_${SLURM_JOB_ID}.tar.gz"
tar -czf "$TARBALL" \
    -C "$(dirname "$OUT")" \
    "$(basename "$OUT")"
log "Tarball: $(du -sh "$TARBALL" | cut -f1) → ${TARBALL}"
log "Done."
