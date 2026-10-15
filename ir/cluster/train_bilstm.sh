#!/bin/bash
#SBATCH --job-name=fp_bilstm
#SBATCH --partition=blaise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/jhdssilva/tcc/ir/logs/bilstm_%j.log
#SBATCH --error=/home/users/jhdssilva/tcc/ir/logs/bilstm_%j.err

set -euo pipefail

CLUSTER_ROOT="/home/users/jhdssilva/tcc/ir"
PYTHON="/home/users/jhdssilva/tcc/miniconda/envs/transformer/bin/python"
SCRIPTS="$CLUSTER_ROOT/scripts"
DATA="$CLUSTER_ROOT/data/translated_dataset.parquet"
OUT="$CLUSTER_ROOT/bilstm_${SLURM_JOB_ID}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$OUT" "$CLUSTER_ROOT/logs"

log "Job ID: ${SLURM_JOB_ID}"
log "Node:   $(hostname)"
log "GPUs:   ${CUDA_VISIBLE_DEVICES:-unset}"
log "Python: $("$PYTHON" --version)"

log "GPU check:"
"$PYTHON" -c "
import torch
print(f'  torch {torch.__version__}  CUDA built={torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

log "Dataset: $(du -sh "$DATA" | cut -f1)"
log "Output:  ${OUT}"

log "Starting BiLSTM training..."
"$PYTHON" "$SCRIPTS/train_bilstm.py" \
    --dataset     "$DATA" \
    --out         "$OUT" \
    --neg-ratio   10 \
    --max-len     512 \
    --emb-dim     64 \
    --hidden-dim  256 \
    --num-layers  2 \
    --dropout     0.3 \
    --epochs      15 \
    --batch-size  512 \
    --lr          1e-3 \
    --seed        42

log "Training complete. Artifacts:"
ls -lh "$OUT"

log "Packaging artifacts..."
TARBALL="$CLUSTER_ROOT/bilstm_${SLURM_JOB_ID}.tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "Tarball: $(du -sh "$TARBALL" | cut -f1) → ${TARBALL}"
log "Done."
