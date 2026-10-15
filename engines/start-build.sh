#!/bin/bash
# Submits, monitors and downloads the instrumented Chromium build.
# Expects build_chromium.slurm.sh in CLUSTER_SRC on the cluster.
#
# Usage:
#   bash engines/start-build.sh            submit a new job and monitor it
#   bash engines/start-build.sh <JOB_ID>   resume monitoring an existing job

set -uo pipefail

CLUSTER_USER="jhdssilva"
CLUSTER_HOST="pcad.inf.ufrgs.br"
CLUSTER_SRC="/home/users/jhdssilva/tcc/chromium/src"
CLUSTER_ARTIFACTS="/home/users/jhdssilva/tcc/artifacts"
LOCAL_OUT="$(cd "$(dirname "$0")/.." && pwd)/navegadores/v8-tcc/out/Release"

POLL_INTERVAL=30    # seconds between status checks

# $HOME instead of ~: bash does not expand ~ inside double quotes,
# so SSH would not find the key.
SSH_OPTS="-i $HOME/.ssh/id_ed25519 \
          -o ServerAliveInterval=20 \
          -o ServerAliveCountMax=5 \
          -o ConnectTimeout=15 \
          -o StrictHostKeyChecking=accept-new"
SSH="ssh ${SSH_OPTS} ${CLUSTER_USER}@${CLUSTER_HOST}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { echo "  │ $*"; }

ssh_run() {
    # Run a command on the cluster with up to 3 attempts and a 10s backoff.
    # stderr stays visible for diagnosis.
    local attempt=0
    while true; do
        if $SSH "$@"; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ $attempt -ge 3 ] && { log "[WARN] SSH failed 3 times, continuing..." >&2; return 1; }
        log "[WARN] SSH failed, attempt ${attempt}/3 in 10s..." >&2
        sleep 10
    done
}

# Preflight: check connectivity first 
log "Checking connectivity to ${CLUSTER_HOST}..."
if ! $SSH "echo ok" > /dev/null 2>/tmp/ssh_preflight_err; then
    log "[ERROR] Could not connect to the cluster. Details:"
    cat /tmp/ssh_preflight_err
    exit 1
fi
log "Connectivity OK."

# Submit or resume 
if [ -n "${1:-}" ]; then
    JOB_ID="$1"
    log "Resuming monitoring of job ${JOB_ID}"
else
    log "Submitting job to the cluster..."
    SBATCH_OUT=$(ssh_run "cd '${CLUSTER_SRC}' && sbatch build_chromium.slurm.sh") || \
        { log "[ERROR] Job submission failed"; exit 1; }

    # "Submitted batch job 12345" → take the last token
    JOB_ID=$(echo "$SBATCH_OUT" | awk '{print $NF}')
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || \
        { log "[ERROR] Invalid job ID: '${JOB_ID}', output: ${SBATCH_OUT}"; exit 1; }

    log "Job submitted: ID=${JOB_ID}"
fi

LOG_FILE="${CLUSTER_SRC}/build_chromium_${JOB_ID}.log"
ERR_FILE="${CLUSTER_SRC}/build_chromium_${JOB_ID}.err"
LAST_LINE=0

log "To resume if the connection drops: bash $0 ${JOB_ID}"
log "Ctrl+C stops monitoring; the job keeps running on the cluster."
echo ""

# Monitoring helpers
get_job_state() {
    local state
    # squeue reports the state while the job is active (PENDING, RUNNING, etc.)
    state=$(ssh_run "squeue -j ${JOB_ID} -h -o '%T' 2>/dev/null" || true)

    if [ -z "${state// /}" ]; then
        # Job left the queue: sacct reports the final state (COMPLETED, FAILED, etc.)
        state=$(ssh_run \
            "sacct -j ${JOB_ID} -n -o State%20 2>/dev/null | head -1 | tr -d ' '" \
            || true)
    fi

    echo "${state:-UNKNOWN}"
}

show_new_log_lines() {
    local total
    total=$(ssh_run "wc -l < '${LOG_FILE}' 2>/dev/null || echo 0" || echo 0)
    total=$(echo "$total" | tr -d ' \n')

    if [ "$total" -gt "$LAST_LINE" ]; then
        ssh_run \
            "sed -n '$((LAST_LINE + 1)),${total}p' '${LOG_FILE}' 2>/dev/null" \
            | while IFS= read -r line; do info "$line"; done \
            || true
        LAST_LINE=$total
    fi
}

show_err_tail() {
    log "Last lines of the error log:"
    ssh_run "tail -20 '${ERR_FILE}' 2>/dev/null" \
        | while IFS= read -r line; do info "$line"; done \
        || true
}

# Monitoring loop 
FINAL_STATE=""
while true; do
    STATE=$(get_job_state)

    case "$STATE" in
        PENDING|CONFIGURING)
            log "Status: PENDING (queued), next check in ${POLL_INTERVAL}s"
            ;;

        RUNNING)
            log "Status: BUILDING"
            show_new_log_lines || log "[WARN] Log unavailable this cycle"
            ;;

        COMPLETED)
            log "Status: COMPLETED ✓"
            show_new_log_lines || true
            FINAL_STATE="COMPLETED"
            break
            ;;

        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
            log "[ERROR] Job ended with state: ${STATE}"
            show_new_log_lines || true
            show_err_tail
            exit 1
            ;;

        UNKNOWN)
            log "[WARN] Unknown state, retrying in ${POLL_INTERVAL}s"
            ;;

        *)
            log "Status: ${STATE}"
            show_new_log_lines || true
            ;;
    esac

    sleep "$POLL_INTERVAL"
done

[ "$FINAL_STATE" = "COMPLETED" ] || exit 1

# Download artifacts 
echo ""
log "Looking for the artifact on the cluster..."
TARBALL_NAME=$(ssh_run \
    "ls -t '${CLUSTER_ARTIFACTS}'/chrome_instrumented_${JOB_ID}*.tar.zst 2>/dev/null \
     | head -1 | xargs -r basename" \
    || true)

[ -n "$TARBALL_NAME" ] || \
    { log "[ERROR] Tarball not found in ${CLUSTER_ARTIFACTS} for job ${JOB_ID}"; exit 1; }

TARBALL_SIZE=$(ssh_run \
    "du -sh '${CLUSTER_ARTIFACTS}/${TARBALL_NAME}' 2>/dev/null | cut -f1" || echo "?")
log "Downloading ${TARBALL_NAME} (${TARBALL_SIZE})..."

rsync \
    -ah \
    --progress \
    --partial \
    -e "ssh ${SSH_OPTS}" \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ARTIFACTS}/${TARBALL_NAME}" \
    /tmp/

log "Extracting to ${LOCAL_OUT}..."
mkdir -p "$LOCAL_OUT"
tar --use-compress-program="zstd -d" -xf "/tmp/${TARBALL_NAME}" -C "$LOCAL_OUT"
chmod +x "${LOCAL_OUT}/chrome"

# Final check 
echo ""
log "Checking instrumentation in the local binary..."
COUNT=$(strings "${LOCAL_OUT}/chrome" | grep -c "\[V8Log\]" || true)
if [ "$COUNT" -gt 0 ]; then
    log "✓ Binary is instrumented (${COUNT} occurrences of [V8Log])"
else
    log "✗ WARNING: instrumentation strings not found in the binary"
fi

log "Done. Binary at: ${LOCAL_OUT}/chrome"
log "Manual test (stderr shows [V8Log] lines):"
log "  ${LOCAL_OUT}/chrome --no-sandbox --js-flags='--single-threaded' https://browserleaks.com/canvas 2>&1 | grep '\\[V8Log\\]' | head -5"
