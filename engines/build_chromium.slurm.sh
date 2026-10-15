#!/bin/bash
#SBATCH --job-name=build_chromium
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=build_chromium_%j.log
#SBATCH --error=build_chromium_%j.err

set -e
JOBS="${SLURM_CPUS_PER_TASK:-16}"
CHROMIUM_ROOT="/home/users/jhdssilva/tcc/chromium/src"
ARTIFACT_DIR="/home/users/jhdssilva/tcc/artifacts"
export PATH="$HOME/tcc/depot_tools:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Copiar interpreter.cc instrumentado 
[ -f ~/tcc/v8/src/interpreter/interpreter.cc ] || \
    { log "[ERRO] interpreter.cc não encontrado"; exit 1; }

cd "$CHROMIUM_ROOT"
cp ~/tcc/v8/src/interpreter/interpreter.cc v8/src/interpreter/interpreter.cc
touch v8/src/interpreter/interpreter.cc   # garante timestamp mais novo que o .o
log "interpreter.cc copiado"

# ── 2. Compilar 
log "Iniciando compilação com ${JOBS} jobs..."
/usr/local/bin/guix shell python -- \
    ninja -C out/Release -j "${JOBS}" chrome
log "Build concluído"

# ── 3. Verificar que a instrumentação está no binário 
COUNT=$(strings out/Release/chrome | grep -c "\[V8Log\]" || true)
log "Strings de instrumentação no binário: ${COUNT}"
[ "$COUNT" -gt 0 ] || { log "[ERRO] Binário não contém instrumentação!"; exit 1; }

# ── 4. Teste de runtime: capturar stderr por 30s com xvfb 
log "Testando instrumentação em runtime..."
STDERR_LOG="/tmp/v8_tcc_test_${SLURM_JOB_ID}.log"

# Xvfb necessário para rodar Chrome headless sem GPU no cluster
Xvfb :99 -screen 0 1280x720x24 &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

out/Release/chrome \
    --no-sandbox \
    --headless=old \
    --disable-gpu \
    --js-flags="--single-threaded" \
    --disable-features=ScriptStreaming \
    "https://browserleaks.com/canvas" \
    2>"$STDERR_LOG" &
CHROME_PID=$!

sleep 30
kill "$CHROME_PID" 2>/dev/null || true
kill "$XVFB_PID"  2>/dev/null || true

TCC_LINES=$(grep -c "\[V8Log\]" "$STDERR_LOG" 2>/dev/null || true)
log "Linhas [V8Log] capturadas: ${TCC_LINES}"
head -5 "$STDERR_LOG" >> build_chromium_"${SLURM_JOB_ID}".log

# ── 5. Empacotar apenas o necessário (não o out/Release inteiro) 
mkdir -p "$ARTIFACT_DIR"
TARBALL="${ARTIFACT_DIR}/chrome_instrumented_${SLURM_JOB_ID}.tar.zst"

log "Compactando artefatos..."
tar \
    --use-compress-program="zstd -T${JOBS} -3" \
    -cf "$TARBALL" \
    -C out/Release \
    chrome \
    chrome_crashpad_handler \
    icudtl.dat \
    v8_context_snapshot.bin \
    resources.pak \
    chrome_100_percent.pak \
    chrome_200_percent.pak \
    locales/ \
    $(ls out/Release/*.so 2>/dev/null | xargs -I{} basename {} || true)

TARBALL_SIZE=$(du -sh "$TARBALL" | cut -f1)
log "Artefato gerado: ${TARBALL} (${TARBALL_SIZE})"
log "Para baixar localmente:"
log "  bash engines/start-build.sh ${SLURM_JOB_ID}"
