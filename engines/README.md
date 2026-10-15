# JavaScript engine instrumentation

Changes made to V8 (Chromium) and SpiderMonkey (Firefox) to dump the bytecodes of every compiled function and to trace anonymous functions.

## Overview

The instrumentation captures, at compile time, the bytecode sequence of each JavaScript function together with metadata that links it to fingerprinting API calls observed at runtime. Each engine writes to stderr with its own prefix:

| Engine | Prefix | Crawler output |
|---|---|---|
| V8 (Chromium) | `[V8Log]` | `data/chromium_v8_bytecodes.jsonl.gz` |
| SpiderMonkey (Firefox) | `[SpiderMonkey Log]` | `data/firefox_sm_bytecodes.jsonl.gz` |

## Patches

| Patch | File changed | Base commit |
|---|---|---|
| `v8-interpreter.patch` | `v8/src/interpreter/interpreter.cc` | V8 `15ab4d3626ed3b62c096f90073cf3d8a6e3c300a` |
| `spidermonkey-bytecode-emitter.patch` | `js/src/frontend/BytecodeEmitter.cpp` | gecko-dev `5836a062726f715fda621338a17b51aff30d0a8c` |

The crawlers expect the source trees under `navegadores/` (gitignored):

```bash
git -C navegadores/v8-tcc/v8 apply ../../../engines/v8-interpreter.patch
git -C navegadores/spidermonkey-tcc/gecko-dev apply ../../../engines/spidermonkey-bytecode-emitter.patch
```

## 1. V8 (`interpreter.cc`)

The instrumentation lives in `DoFinalizeJobImpl`, which Ignition calls when it finishes compiling a function, right after the `BytecodeArray` is generated. Functions without a `Script` (built-ins, extensions, WASM) are skipped.

**Stable ID.** Each function gets a deterministic `injected_id`: FNV-1a 64-bit over

```
"v8/" + script_url + "/" + start_pos + "/" + end_pos
```

where `start_pos`/`end_pos` are character offsets in the script source. FNV-1a is used instead of `std::hash`, which many compilers randomize per process.

**Output.** One JSON line per function, serialized by a static mutex so that concurrent compiler threads do not interleave:

```json
{"injected_id":"fn_12345678901234567","script_url":"https://example.com/bundle.js","script_id":42,"start_pos":1024,"end_pos":2048,"func_name":"canvasFingerprint","bytecodes":"LdaConstant,Star0,CallProperty0,..."}
```

| Field | Description |
|---|---|
| `injected_id` | FNV-1a 64-bit hash with `fn_` prefix; stable across sessions |
| `script_url` | Script URL (the page URL for inline scripts) |
| `script_id` | V8 internal script ID for this session |
| `start_pos`, `end_pos` | Function offsets in the script source |
| `func_name` | AST name; `"(anonymous)"` for unnamed functions |
| `bytecodes` | Comma-separated opcodes |

## 2. SpiderMonkey (`BytecodeEmitter.cpp`)

Each anonymous function gets the display name `anon_injected_<hash>`, where the hash (`std::hash`) covers the script URL and the function position. The display name shows up in `Error.stack`, so the extension captures the same identifier as the bytecode log, and the Firefox join is purely name-based.

Each compiled function is written to stderr as a text block:

```
[SpiderMonkey Log]
Script URL: https://example.com/bundle.js
Internal Index: 7
Start: 1024
End: 2048
Function name: canvasFingerprint
Bytecode: [FunctionThis, SetLocal, Pop, GetGName, ...]
========================================
```

`parse_spidermonkey_logs()` in `crawlers/firefox-crawler.py` turns these blocks into the same JSONL as Chromium (without `start_pos`/`end_pos`) and drops internal URLs (`resource://`, `chrome://`, `moz-extension://`, ...).

## 3. Fingerprinting API capture (shared extension)

`extension/` is loaded in both browsers. `content.js` runs in the MAIN world at `document_start` in every frame and hooks Canvas, Canvas Font, AudioContext, WebRTC, WebGL, `screen` and `navigator` APIs (full list in `content.js`). It also normalizes Firefox-only properties (`navigator.oscpu`, `buildID`, `mozInnerScreenX`, ...) so fingerprinting libraries do not skip collection after detecting the browser.

Each call records one entry per HTTP frame in the stack trace (V8 and Gecko formats):

```json
{"api":"canvas.toDataURL","funcName":"canvasFingerprint","scriptUrl":"https://example.com/bundle.js","line":42,"col":9}
```

The crawlers read `window.__fpCalls` after each page.

## 4. Building

### 4.1 Chromium (PCAD/UFRGS cluster)

The instrumented binary was built on the PCAD cluster with SLURM. The scripts hardcode the account and paths used there; adjust `CLUSTER_USER` and the paths before using them. `build_chromium.slurm.sh` must be in `CLUSTER_SRC` on the cluster.

```bash
# 1. Copy the patched interpreter.cc to the cluster
scp navegadores/v8-tcc/v8/src/interpreter/interpreter.cc \
    <user>@pcad.inf.ufrgs.br:~/tcc/v8/src/interpreter/interpreter.cc

# 2. Submit the job and monitor it (downloads the binary when done)
bash engines/start-build.sh

# Resume monitoring an existing job
bash engines/start-build.sh <JOB_ID>
```

The binary is extracted to `navegadores/v8-tcc/out/Release/chrome`. To check the instrumentation:

```bash
strings navegadores/v8-tcc/out/Release/chrome | grep -c '\[V8Log\]'   # must be > 0
```

### 4.2 Firefox (local)

```bash
cd navegadores/spidermonkey-tcc/gecko-dev
python3.12 mach build browser
```

The build creates symlinks in `dist/bin/` with the absolute source path. If the tree is moved after building, add a symlink at the old path pointing to the new one, or rebuild.

## 5. Validating

```bash
navegadores/v8-tcc/out/Release/chrome --no-sandbox --js-flags='--single-threaded' \
  https://browserleaks.com/canvas 2>&1 | grep '^\[V8Log\]' | head -3

navegadores/spidermonkey-tcc/gecko-dev/obj-x86_64-pc-linux-gnu/dist/bin/firefox \
  --headless --no-remote --profile /tmp/ff-test \
  https://browserleaks.com/canvas 2>&1 | grep -A6 '^\[SpiderMonkey Log\]' | head -20
```

## 6. Known limitations

| Limitation | Detail |
|---|---|
| Lazy compilation | Bytecodes are only emitted when a function is compiled; lazily loaded code appears only if the page runs it during the crawl. |
| Web Workers | The extension does not hook Workers, so fingerprinting done inside Workers is not captured. |
| `eval` / `new Function` | Runtime-generated functions get an ID based on the position of the `eval`, not of the generated code. |
| Parallel compilation (V8) | The mutex serializes writes and can become a bottleneck; mitigated with `--single-threaded`. |
| Inline scripts | Report the page URL; `start_pos`/`end_pos` are relative to the whole HTML. |
