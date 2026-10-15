#!/usr/bin/env python3
"""
Run all micro-scripts through both instrumented browsers and save bytecode pairs.

For each JS file in ir/microscripts/, launches Chromium (V8) and Firefox (SM)
once per engine, navigates to every script, and saves per-script records to
ir/microscript_logs/<stem>.json with shape:
  {"script": "<stem>", "v8": [...records...], "sm": [...records...]}

Usage:
  python ir/run_microscripts.py           # run all scripts
  python ir/run_microscripts.py 09        # run scripts whose stem contains '09'
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FFService
from selenium.webdriver.firefox.options import Options as FFOptions

# ── Paths ──────────────────────────────────────────────────────────────────────
TCC_ROOT    = Path(__file__).resolve().parent.parent.parent
CHROMIUM    = TCC_ROOT / "navegadores/v8-tcc/out/Release/chrome"
FIREFOX_BIN = (
    TCC_ROOT
    / "navegadores/spidermonkey-tcc/gecko-dev"
    / "obj-x86_64-pc-linux-gnu/dist/bin/firefox"
)
GECKODRIVER = Path("/usr/local/bin/geckodriver")
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "microscripts"
LOGS_DIR    = Path(__file__).resolve().parent.parent / "microscript_logs"
GK_LOG      = LOGS_DIR / "geckodriver_raw.log"

CDP_PORT  = 9299   # separate from main crawler (9222) to avoid conflicts
PAGE_WAIT = 2.0    # seconds after domcontentloaded for compilation to finish

V8_SKIP_PREFIXES = ("extensions::", "chrome://", "v8/", "unknown")
SM_SKIP_PREFIXES = (
    "resource://", "chrome://", "about:", "moz-src:",
    "moz-extension://", "moz-nullprincipal:", "<anonymous code>", "data:",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _killpg(pgid: int, proc: subprocess.Popen) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _wait_cdp(port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def _make_html(js_code: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        f"<body><script>\n{js_code}\n</script></body></html>"
    )


# ── V8 / Chromium ──────────────────────────────────────────────────────────────

def run_v8(scripts: list[tuple[str, Path]]) -> dict[str, list[dict]]:
    """Launch Chromium once, navigate each script, collect V8 bytecodes incrementally."""
    user_data = Path(tempfile.mkdtemp(prefix="ir_chromium_"))
    flags = [
        str(CHROMIUM),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={user_data}",
        "--no-sandbox", "--no-first-run", "--no-restore-state",
        "--disable-features=ScriptStreaming",
        "--js-flags=--single-threaded",
        "--headless=new", "--ozone-platform=x11",
        "--disable-dev-shm-usage", "--disable-gpu", "--disk-cache-size=0",
        "about:blank",
    ]
    v8_lines: list[str] = []
    proc = subprocess.Popen(
        flags, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    pgid = os.getpgid(proc.pid)

    def _read_stderr():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.startswith("[V8Log] "):
                v8_lines.append(line[8:])

    threading.Thread(target=_read_stderr, daemon=True).start()

    if not _wait_cdp(CDP_PORT):
        _killpg(pgid, proc)
        shutil.rmtree(user_data, ignore_errors=True)
        raise RuntimeError("Chromium CDP did not open — check the binary")

    results: dict[str, list[dict]] = {stem: [] for stem, _ in scripts}

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()

            for stem, html_path in scripts:
                print(f"  [V8] {stem}")
                idx_before = len(v8_lines)
                page = ctx.new_page()
                try:
                    page.goto(
                        f"file://{html_path}",
                        wait_until="domcontentloaded",
                        timeout=10_000,
                    )
                    time.sleep(PAGE_WAIT)
                except Exception as e:
                    print(f"    [WARN] {e}")
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                for line in v8_lines[idx_before:]:
                    try:
                        raw = json.loads(line)
                        url = raw.get("script_url", "")
                        if any(url.startswith(p) for p in V8_SKIP_PREFIXES):
                            continue
                        results[stem].append({
                            "function_name": raw["func_name"],
                            "injected_id":   raw["injected_id"],
                            "script_url":    url,
                            "start_pos":     raw["start_pos"],
                            "end_pos":       raw["end_pos"],
                            "bytecodes":     raw["bytecodes"],
                        })
                    except Exception:
                        pass

            try:
                browser.disconnect()
            except Exception:
                pass
    finally:
        _killpg(pgid, proc)
        shutil.rmtree(user_data, ignore_errors=True)

    return results


# ── SM / Firefox ───────────────────────────────────────────────────────────────

def run_sm(scripts: list[tuple[str, Path]]) -> dict[str, list[dict]]:
    """Launch Firefox once, navigate each script, parse geckodriver log grouped by URL."""
    GK_LOG.write_bytes(b"")   # truncate before session starts

    ff_opts = FFOptions()
    ff_opts.binary_location = str(FIREFOX_BIN)
    ff_opts.add_argument("--headless")
    ff_opts.page_load_strategy = "none"
    ff_service = FFService(executable_path=str(GECKODRIVER), log_output=str(GK_LOG))

    driver = webdriver.Firefox(service=ff_service, options=ff_opts)
    driver.set_page_load_timeout(10)

    url_to_stem: dict[str, str] = {
        f"file://{html_path}": stem for stem, html_path in scripts
    }

    for stem, html_path in scripts:
        print(f"  [SM] {stem}")
        try:
            driver.get(f"file://{html_path}")
        except Exception as e:
            print(f"    [WARN] {e}")
        time.sleep(PAGE_WAIT)

    try:
        driver.quit()
    except Exception:
        pass
    time.sleep(0.5)   # let geckodriver flush log to disk

    results: dict[str, list[dict]] = {stem: [] for stem, _ in scripts}
    with open(GK_LOG, "r", encoding="utf-8", errors="replace") as f:
        it = iter(f)
        for line in it:
            if line.strip() != "[SpiderMonkey Log]":
                continue
            try:
                script_url = next(it).split("Script URL: ", 1)[1].strip()
                script_id  = next(it).split("Internal Index: ", 1)[1].strip()
                func_name  = next(it).split("Function name: ", 1)[1].strip()
                bytecodes  = next(it).split("Bytecode: [", 1)[1].rstrip().rstrip("]").strip()
            except StopIteration:
                break
            except IndexError:
                continue
            if any(script_url.startswith(p) for p in SM_SKIP_PREFIXES):
                continue
            stem = url_to_stem.get(script_url)
            if stem is None:
                continue
            results[stem].append({
                "function_name": func_name,
                "injected_id":   func_name,
                "script_url":    script_url,
                "script_id":     script_id,
                "bytecodes":     bytecodes,
            })

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    filter_arg = sys.argv[1] if len(sys.argv) > 1 else None
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    js_files = sorted(SCRIPTS_DIR.glob("*.js"))
    if filter_arg:
        js_files = [f for f in js_files if filter_arg in f.stem]
    if not js_files:
        sys.exit(f"No scripts found in {SCRIPTS_DIR} matching {filter_arg!r}")

    for binary, label in [(CHROMIUM, "Chromium"), (FIREFOX_BIN, "Firefox")]:
        if not binary.exists():
            sys.exit(f"[ERROR] {label} binary not found: {binary}")

    # Each script gets a distinct temp directory so its file:// URL is unique
    tmp_root = Path(tempfile.mkdtemp(prefix="ir_microscripts_"))
    scripts: list[tuple[str, Path]] = []
    for js_path in js_files:
        stem = js_path.stem
        html_dir = tmp_root / stem
        html_dir.mkdir()
        html_path = html_dir / "page.html"
        html_path.write_text(_make_html(js_path.read_text(encoding="utf-8")))
        scripts.append((stem, html_path))

    print(f"\n{'='*60}")
    print(f"Running {len(scripts)} scripts through V8 (Chromium)...")
    print(f"{'='*60}")
    v8_results = run_v8(scripts)

    print(f"\n{'='*60}")
    print(f"Running {len(scripts)} scripts through SpiderMonkey (Firefox)...")
    print(f"{'='*60}")
    sm_results = run_sm(scripts)

    print(f"\n{'='*60}")
    print("Saving results...")
    for stem, _ in scripts:
        out = {
            "script":  stem,
            "v8":      v8_results[stem],
            "sm":      sm_results[stem],
        }
        out_path = LOGS_DIR / f"{stem}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        v8_n = len(v8_results[stem])
        sm_n = len(sm_results[stem])
        status = "OK" if v8_n > 0 and sm_n > 0 else "WARN: missing data"
        print(f"  {stem}: V8={v8_n}  SM={sm_n}  [{status}]")

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nDone. Logs → {LOGS_DIR}")
    print("Next step: ir/build_equivalence_dict.py")


if __name__ == "__main__":
    main()
