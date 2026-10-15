#!/bin/bash
# pipeline_local.sh
# Submete, monitora e baixa o build do Chromium instrumentado.
#
# Uso:
#   bash pipeline_local.sh           — submete novo job e monitora
#   bash pipeline_local.sh <JOB_ID>  — retoma monitoramento de job existente

set -uo pipefail

CLUSTER_USER="jhdssilva"
CLUSTER_HOST="pcad.inf.ufrgs.br"
CLUSTER_SRC="/home/users/jhdssilva/tcc/chromium/src"
CLUSTER_ARTIFACTS="/home/users/jhdssilva/tcc/artifacts"
LOCAL_OUT="$(cd "$(dirname "$0")/.." && pwd)/navegadores/v8-tcc/out/Release"

POLL_INTERVAL=30    # segundos entre cada verificação de status

# $HOME em vez de ~ : dentro de aspas duplas, ~ não é expandido pelo bash,
# mas $HOME sim. Sem isso, o SSH não encontra a chave.
SSH_OPTS="-i $HOME/.ssh/id_ed25519 \
          -o ServerAliveInterval=20 \
          -o ServerAliveCountMax=5 \
          -o ConnectTimeout=15 \
          -o StrictHostKeyChecking=accept-new"
SSH="ssh ${SSH_OPTS} ${CLUSTER_USER}@${CLUSTER_HOST}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { echo "  │ $*"; }

ssh_run() {
    # Executa um comando no cluster com até 3 tentativas e backoff de 10s.
    # stderr sempre visível para facilitar diagnóstico.
    local attempt=0
    while true; do
        if $SSH "$@"; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ $attempt -ge 3 ] && { log "[WARN] SSH falhou 3x — continuando..." >&2; return 1; }
        log "[WARN] SSH falhou, tentativa ${attempt}/3 em 10s..." >&2
        sleep 10
    done
}

# Preflight: testar conectividade antes de qualquer coisa 
log "Testando conectividade com ${CLUSTER_HOST}..."
if ! $SSH "echo ok" > /dev/null 2>/tmp/ssh_preflight_err; then
    log "[ERRO] Não foi possível conectar ao cluster. Detalhes:"
    cat /tmp/ssh_preflight_err
    exit 1
fi
log "Conectividade OK."

# Submeter ou retomar 
if [ -n "${1:-}" ]; then
    JOB_ID="$1"
    log "Retomando monitoramento do job ${JOB_ID}"
else
    log "Submetendo job no cluster..."
    SBATCH_OUT=$(ssh_run "cd '${CLUSTER_SRC}' && sbatch cluster_pipeline.sh") || \
        { log "[ERRO] Falha ao submeter job"; exit 1; }

    # "Submitted batch job 12345" → pega o último token
    JOB_ID=$(echo "$SBATCH_OUT" | awk '{print $NF}')
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || \
        { log "[ERRO] Job ID inválido: '${JOB_ID}' — output: ${SBATCH_OUT}"; exit 1; }

    log "Job submetido: ID=${JOB_ID}"
fi

LOG_FILE="${CLUSTER_SRC}/build_chromium_${JOB_ID}.log"
ERR_FILE="${CLUSTER_SRC}/build_chromium_${JOB_ID}.err"
LAST_LINE=0

log "Para retomar caso a conexão caia: bash $0 ${JOB_ID}"
log "Ctrl+C interrompe o monitoramento — o job continua no cluster."
echo ""

# Funções de monitoramento
get_job_state() {
    local state
    # squeue retorna o estado enquanto o job está ativo (PENDING, RUNNING, etc.)
    state=$(ssh_run "squeue -j ${JOB_ID} -h -o '%T' 2>/dev/null" || true)

    if [ -z "${state// /}" ]; then
        # Job saiu da fila: sacct traz o estado final (COMPLETED, FAILED, etc.)
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
    log "Últimas linhas do log de erro:"
    ssh_run "tail -20 '${ERR_FILE}' 2>/dev/null" \
        | while IFS= read -r line; do info "$line"; done \
        || true
}

# Loop de monitoramento 
FINAL_STATE=""
while true; do
    STATE=$(get_job_state)

    case "$STATE" in
        PENDING|CONFIGURING)
            log "Status: PENDENTE (na fila) — próxima verificação em ${POLL_INTERVAL}s"
            ;;

        RUNNING)
            log "Status: COMPILANDO "
            show_new_log_lines || log "[WARN] Log inacessível neste ciclo"
            ;;

        COMPLETED)
            log "Status: CONCLUÍDO ✓"
            show_new_log_lines || true
            FINAL_STATE="COMPLETED"
            break
            ;;

        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL)
            log "[ERRO] Job terminou com estado: ${STATE}"
            show_new_log_lines || true
            show_err_tail
            exit 1
            ;;

        UNKNOWN)
            log "[WARN] Estado desconhecido — tentando novamente em ${POLL_INTERVAL}s"
            ;;

        *)
            log "Status: ${STATE}"
            show_new_log_lines || true
            ;;
    esac

    sleep "$POLL_INTERVAL"
done

[ "$FINAL_STATE" = "COMPLETED" ] || exit 1

# Download dos artefatos 
echo ""
log "Procurando artefato no cluster..."
TARBALL_NAME=$(ssh_run \
    "ls -t '${CLUSTER_ARTIFACTS}'/chrome_instrumented_${JOB_ID}*.tar.zst 2>/dev/null \
     | head -1 | xargs -r basename" \
    || true)

[ -n "$TARBALL_NAME" ] || \
    { log "[ERRO] Tarball não encontrado em ${CLUSTER_ARTIFACTS} para job ${JOB_ID}"; exit 1; }

TARBALL_SIZE=$(ssh_run \
    "du -sh '${CLUSTER_ARTIFACTS}/${TARBALL_NAME}' 2>/dev/null | cut -f1" || echo "?")
log "Baixando ${TARBALL_NAME} (${TARBALL_SIZE})..."

rsync \
    -ah \
    --progress \
    --partial \
    -e "ssh ${SSH_OPTS}" \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_ARTIFACTS}/${TARBALL_NAME}" \
    /tmp/

log "Extraindo para ${LOCAL_OUT}..."
mkdir -p "$LOCAL_OUT"
tar --use-compress-program="zstd -d" -xf "/tmp/${TARBALL_NAME}" -C "$LOCAL_OUT"
chmod +x "${LOCAL_OUT}/chrome"

# Verificação final 
echo ""
log "Verificando instrumentação no binário local..."
COUNT=$(strings "${LOCAL_OUT}/chrome" | grep -c "\[V8Log\]" || true)
if [ "$COUNT" -gt 0 ]; then
    log "✓ Binário instrumentado corretamente (${COUNT} ocorrências de [V8Log])"
else
    log "✗ AVISO: strings de instrumentação não encontradas no binário"
fi

log "Pronto. Binário disponível em: ${LOCAL_OUT}/chrome"
log "Para testar manualmente (stderr mostrará [V8Log] lines):"
log "  ${LOCAL_OUT}/chrome --no-sandbox --js-flags='--single-threaded' https://browserleaks.com/canvas 2>&1 | grep '\\[V8Log\\]' | head -5"
