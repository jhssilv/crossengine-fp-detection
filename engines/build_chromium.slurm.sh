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

# ── 1. Copy instrumented interpreter.cc 
[ -f ~/tcc/v8/src/interpreter/interpreter.cc ] || \
    { log "[ERROR] interpreter.cc not found"; exit 1; }

cd "$CHROMIUM_ROOT"
cp ~/tcc/v8/src/interpreter/interpreter.cc v8/src/interpreter/interpreter.cc
touch v8/src/interpreter/interpreter.cc   # make sure it is newer than the .o
log "interpreter.cc copied"

# ── 2. Build 
log "Building with ${JOBS} jobs..."
/usr/local/bin/guix shell python -- \
    ninja -C out/Release -j "${JOBS}" chrome
log "Build finished"

# ── 3. Check the instrumentation is in the binary 
COUNT=$(strings out/Release/chrome | grep -c "\[V8Log\]" || true)
log "Instrumentation strings in binary: ${COUNT}"
[ "$COUNT" -gt 0 ] || { log "[ERROR] Binary has no instrumentation!"; exit 1; }

# ── 4. Runtime test: capture stderr for 30s under xvfb 
log "Testing instrumentation at runtime..."
STDERR_LOG="/tmp/v8_tcc_test_${SLURM_JOB_ID}.log"

# Xvfb is needed to run headless Chrome without a GPU on the cluster
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
log "[V8Log] lines captured: ${TCC_LINES}"
head -5 "$STDERR_LOG" >> build_chromium_"${SLURM_JOB_ID}".log

# ── 5. Package only what is needed (not the whole out/Release) 
mkdir -p "$ARTIFACT_DIR"
TARBALL="${ARTIFACT_DIR}/chrome_instrumented_${SLURM_JOB_ID}.tar.zst"

log "Compressing artifacts..."
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
log "Artifact: ${TARBALL} (${TARBALL_SIZE})"
log "To download locally:"
log "  bash engines/start-build.sh ${SLURM_JOB_ID}"
