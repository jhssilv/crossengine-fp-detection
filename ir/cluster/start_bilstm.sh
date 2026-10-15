#!/bin/bash
# Submit, monitor and download the BiLSTM training job on PCAD blaise nodes.
#
# Usage:
#   bash ir/cluster/start_bilstm.sh           — upload + submit + monitor
#   bash ir/cluster/start_bilstm.sh <JOB_ID>  — resume monitoring existing job

set -uo pipefail

CLUSTER_USER="jhdssilva"
CLUSTER_HOST="pcad.inf.ufrgs.br"
CLUSTER_ROOT="/home/users/jhdssilva/tcc/ir"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # tcc/ root
LOCAL_OUT="${LOCAL_ROOT}/ir/bilstm_model"

POLL_INTERVAL=60

SSH_OPTS="-i $HOME/.ssh/id_ed25519 -o ServerAliveInterval=20 -o ServerAliveCountMax=5 -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
SSH="ssh $SSH_OPTS ${CLUSTER_USER}@${CLUSTER_HOST}"
RSYNC_SSH="ssh $SSH_OPTS"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { echo "  │ $*"; }

ssh_run() {
    local attempt=0
    while true; do
        if $SSH "$@"; then return 0; fi
        attempt=$((attempt + 1))
        [ $attempt -ge 3 ] && { log "[WARN] SSH failed 3× — continuing..." >&2; return 1; }
        log "[WARN] SSH failed, retry ${attempt}/3 in 10s..." >&2
        sleep 10
    done
}

log "Testing connectivity to ${CLUSTER_HOST}..."
ssh_run "echo ok" > /dev/null
log "OK."

if [ -n "${1:-}" ]; then
    JOB_ID="$1"
    log "Resuming monitoring of job ${JOB_ID}"
else
    log "Creating cluster directories..."
    ssh_run "mkdir -p ${CLUSTER_ROOT}/{scripts,data,logs,cluster}"

    log "Uploading train_bilstm.py..."
    rsync -az --progress \
        -e "$RSYNC_SSH" \
        "${LOCAL_ROOT}/ir/scripts/train_bilstm.py" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ROOT}/scripts/train_bilstm.py"

    log "Uploading SLURM script..."
    rsync -az --progress \
        -e "$RSYNC_SSH" \
        "${LOCAL_ROOT}/ir/cluster/train_bilstm.sh" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ROOT}/cluster/train_bilstm.sh"

    # Reuse dataset already on cluster from transformer experiments if present
    DATASET_REMOTE="${CLUSTER_ROOT}/data/translated_dataset.parquet"
    TRANSFORMER_DATASET="/home/users/jhdssilva/tcc/transformer/data/translated_dataset.parquet"
    REMOTE_SIZE=$(ssh_run "stat -c%s '${DATASET_REMOTE}' 2>/dev/null || echo 0" || echo 0)
    LOCAL_SIZE=$(stat -c%s "${LOCAL_ROOT}/ir/translated_dataset.parquet" 2>/dev/null || echo 0)

    if [ "$REMOTE_SIZE" -eq "$LOCAL_SIZE" ] 2>/dev/null && [ "$REMOTE_SIZE" -gt 0 ]; then
        log "Dataset already on cluster — skipping upload."
    else
        # Try symlinking from transformer data first (saves bandwidth)
        TRANSFORMER_SIZE=$(ssh_run "stat -c%s '${TRANSFORMER_DATASET}' 2>/dev/null || echo 0" || echo 0)
        if [ "$TRANSFORMER_SIZE" -eq "$LOCAL_SIZE" ] 2>/dev/null && [ "$TRANSFORMER_SIZE" -gt 0 ]; then
            log "Symlinking dataset from transformer data (already on cluster)..."
            ssh_run "ln -sf '${TRANSFORMER_DATASET}' '${DATASET_REMOTE}'"
        else
            log "Uploading dataset (~$(du -sh "${LOCAL_ROOT}/ir/translated_dataset.parquet" | cut -f1))..."
            rsync -az --progress --partial \
                -e "$RSYNC_SSH" \
                "${LOCAL_ROOT}/ir/translated_dataset.parquet" \
                "${CLUSTER_USER}@${CLUSTER_HOST}:${DATASET_REMOTE}"
        fi
    fi

    log "Submitting BiLSTM training job..."
    SBATCH_OUT=$(ssh_run "sbatch ${CLUSTER_ROOT}/cluster/train_bilstm.sh") || \
        { log "[ERROR] Failed to submit job"; exit 1; }
    JOB_ID=$(echo "$SBATCH_OUT" | awk '{print $NF}')
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || \
        { log "[ERROR] Invalid job ID: '${JOB_ID}' — output: ${SBATCH_OUT}"; exit 1; }
    log "Job submitted: ID=${JOB_ID}"
fi

LOG_FILE="${CLUSTER_ROOT}/logs/bilstm_${JOB_ID}.log"
ERR_FILE="${CLUSTER_ROOT}/logs/bilstm_${JOB_ID}.err"
LAST_LINE=0

log "To resume if connection drops: bash $0 ${JOB_ID}"
log "Ctrl+C stops monitoring — job keeps running on cluster."
echo ""

get_job_state() {
    local state
    state=$(ssh_run "squeue -j ${JOB_ID} -h -o '%T' 2>/dev/null" || true)
    if [ -z "${state// /}" ]; then
        state=$(ssh_run "sacct -j ${JOB_ID} -n -o State%20 2>/dev/null | head -1 | tr -d ' '" || true)
    fi
    echo "${state:-UNKNOWN}"
}

show_new_log_lines() {
    local total
    total=$(ssh_run "wc -l < '${LOG_FILE}' 2>/dev/null || echo 0" || echo 0)
    total=$(echo "$total" | tr -d ' \n')
    if [ "$total" -gt "$LAST_LINE" ]; then
        ssh_run "sed -n '$((LAST_LINE + 1)),${total}p' '${LOG_FILE}' 2>/dev/null" \
            | while IFS= read -r line; do info "$line"; done || true
        LAST_LINE=$total
    fi
}

FINAL_STATE=""
while true; do
    STATE=$(get_job_state)
    case "$STATE" in
        PENDING|CONFIGURING)
            log "Status: PENDING — retry in ${POLL_INTERVAL}s" ;;
        RUNNING)
            log "Status: RUNNING ──────────────────────────────────────"
            show_new_log_lines || true ;;
        COMPLETED)
            log "Status: COMPLETED ✓"
            show_new_log_lines || true
            FINAL_STATE="COMPLETED"
            break ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
            log "[ERROR] Job ended with state: ${STATE}"
            show_new_log_lines || true
            log "Last error lines:"
            ssh_run "tail -20 '${ERR_FILE}' 2>/dev/null" | while IFS= read -r l; do info "$l"; done || true
            exit 1 ;;
        UNKNOWN)
            log "[WARN] Unknown state — retrying in ${POLL_INTERVAL}s" ;;
        *)
            log "Status: ${STATE}"
            show_new_log_lines || true ;;
    esac
    sleep "$POLL_INTERVAL"
done

[ "$FINAL_STATE" = "COMPLETED" ] || exit 1

echo ""
log "Downloading artifacts..."
TARBALL_NAME=$(ssh_run \
    "ls -t '${CLUSTER_ROOT}'/bilstm_${JOB_ID}*.tar.gz 2>/dev/null | head -1 | xargs -r basename" \
    || true)

[ -n "$TARBALL_NAME" ] || \
    { log "[ERROR] Tarball not found for job ${JOB_ID}"; exit 1; }

TARBALL_SIZE=$(ssh_run "du -sh '${CLUSTER_ROOT}/${TARBALL_NAME}' 2>/dev/null | cut -f1" || echo "?")
log "Downloading ${TARBALL_NAME} (${TARBALL_SIZE})..."

rsync -ah --progress --partial \
    -e "$RSYNC_SSH" \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ROOT}/${TARBALL_NAME}" \
    /tmp/

log "Extracting to ${LOCAL_OUT}..."
mkdir -p "$LOCAL_OUT"
tar -xzf "/tmp/${TARBALL_NAME}" -C "$LOCAL_OUT" --strip-components=1

log "Done. Artifacts at: ${LOCAL_OUT}"
cat "${LOCAL_OUT}/metrics.json" 2>/dev/null | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'  F1={m[\"f1\"]:.4f}  P={m[\"precision\"]:.4f}  R={m[\"recall\"]:.4f}  threshold={m[\"threshold\"]:.2f}')
" 2>/dev/null || true
