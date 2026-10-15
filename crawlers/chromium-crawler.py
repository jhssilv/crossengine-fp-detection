"""
Chromium crawler com captura de traces V8 + fingerprinting APIs.

Idempotente: registra URLs concluídas em DONE_FILE. Relançar o script
retoma a partir da última URL pendente, sem reprocessar sites já visitados.

Fluxo por site:
  1. Lança o Chromium customizado (subprocess captura stderr → v8_lines)
  2. Conecta via Playwright over CDP (sem gerenciar o processo)
  3. Para cada URL: navega, simula interação, coleta window.__fpCalls
     → appenda fp_calls imediatamente
     → drena novas linhas v8_lines → appenda bytecodes ao disco
     → marca site como concluído em DONE_FILE
  4. O post_processor.py faz o join offline
"""
import argparse
import gzip
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

# ── Paths ─────────────────────────────────────────────────────────────────────
TCC_ROOT     = Path(__file__).resolve().parent.parent
CHROMIUM_BIN = TCC_ROOT / "navegadores" / "v8-tcc" / "out" / "Release" / "chrome"
EXTENSION    = TCC_ROOT / "extension"
LOGS_DIR     = TCC_ROOT / "data"
TRANCO_FILE  = TCC_ROOT / "crawlers" / "tranco_top10k.csv"

V8_LOGS_FILE  = LOGS_DIR / "chromium_v8_bytecodes.jsonl.gz"
FP_CALLS_FILE = LOGS_DIR / "chromium_fp_calls.jsonl"
DONE_FILE     = LOGS_DIR / "chromium_done.txt"

# ── Config ────────────────────────────────────────────────────────────────────
CRAWL_LIMIT          = 10_000
CDP_PORT             = 9222
DWELL_SECONDS        = 4
FP_WAIT_EXTRA        = 2
PAGE_TIMEOUT         = 15_000  # ms (Playwright)
CHROME_RESTART_EVERY = 10      # restart Chrome every N sites to clean up open pages
SITE_TIMEOUT_S       = 50      # hard wall-clock limit per site (covers sleeps too)

V8_INTERNAL_PREFIXES = (
    "extensions::",
    "chrome://",
    "v8/",
    "unknown",
)

def _killpg(pgid: int, proc: subprocess.Popen) -> None:
    """Kill the entire Chrome process group (browser + renderers + GPU process).

    When Chrome is OOM-killed by the OS, only the browser process dies; renderer
    and GPU child processes become orphans and accumulate in RAM.  Killing by
    process group ensures every Chrome subprocess is reaped at once.
    """
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already dead
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _make_watchdog(timeout: float, proc_ref: list, pgid_ref: list):
    """Kill Chrome after timeout; if Python is still stuck 15s later, kill self.

    Returns (timer, fired_event, cancel_nuke).
    - fired_event: set when watchdog fires; skip page.close() if set.
    - cancel_nuke: call this when crawl_site() returns after a watchdog kill so the
      nuclear timer doesn't fire mid-drain_v8_lines and corrupt the gzip log.
    """
    fired    = threading.Event()
    nuke_ref = [None]

    def _kill():
        fired.set()
        print(f"\n  [WATCHDOG] >{timeout:.0f}s — matando Chrome para desbloquear")
        _killpg(pgid_ref[0], proc_ref[0])
        print("  [WATCHDOG] Chrome killed. Nuclear timer em 15s caso crawl_site não retorne.")

        def _self_kill():
            import sys as _sys
            print("  [WATCHDOG NUCLEAR] crawl_site não retornou após kill — reiniciando Python",
                  flush=True)
            _sys.stderr.flush()
            os.kill(os.getpid(), 9)

        nuke = threading.Timer(15, _self_kill)
        nuke.daemon = True
        nuke_ref[0] = nuke
        nuke.start()

    def cancel_nuke():
        if nuke_ref[0] is not None:
            nuke_ref[0].cancel()

    t = threading.Timer(timeout, _kill)
    t.daemon = True
    return t, fired, cancel_nuke

# ── Tranco + progresso ────────────────────────────────────────────────────────

def load_target_sites() -> list[str]:
    urls = []
    with open(TRANCO_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                urls.append(f"https://{parts[1]}")
            if len(urls) >= CRAWL_LIMIT:
                break
    return urls


def load_completed() -> set[str]:
    if not DONE_FILE.exists():
        return set()
    return set(DONE_FILE.read_text(encoding="utf-8").splitlines())


def mark_completed(url: str) -> None:
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def _v8_key(injected_id: str) -> int:
    """FNV-1a 64-bit hash of injected_id — replaces string set (~10x smaller)."""
    h = 14695981039346656037
    for c in injected_id.encode():
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h



# ── V8 log drain ──────────────────────────────────────────────────────────────

def _make_stderr_writer(proc: subprocess.Popen, seen_ids: set[int]
                        ) -> tuple[list[int], threading.Thread]:
    """Start a daemon thread that writes V8 log lines directly to disk as they arrive.

    Returns (bc_counter, thread).
    bc_counter = [written, filtered] — incremented by the thread; read by main for stats.
    Writing directly avoids accumulating millions of lines in a list between sites.
    """
    bc_counter = [0, 0]

    def _run():
        with gzip.open(V8_LOGS_FILE, "at", encoding="utf-8") as gz:
            for raw_bytes in iter(proc.stderr.readline, b""):
                text = raw_bytes.decode("utf-8", errors="replace").rstrip()
                if not text.startswith("[V8Log] "):
                    continue
                try:
                    raw = json.loads(text[8:])
                except json.JSONDecodeError:
                    continue
                url = raw.get("script_url", "")
                if (any(url.startswith(p) for p in V8_INTERNAL_PREFIXES)
                        or not url.startswith("http")):
                    bc_counter[1] += 1
                    continue
                key = _v8_key(raw["injected_id"])
                if key in seen_ids:
                    bc_counter[1] += 1
                    continue
                seen_ids.add(key)
                record = {
                    "browser":       "Chromium",
                    "injected_id":   raw.get("injected_id", key),
                    "script_url":    url,
                    "script_id":     raw.get("script_id", 0),
                    "function_name": raw.get("func_name", ""),
                    "start_pos":     raw.get("start_pos", 0),
                    "end_pos":       raw.get("end_pos", 0),
                    "bytecodes":     raw.get("bytecodes", ""),
                }
                json.dump(record, gz)
                gz.write("\n")
                bc_counter[0] += 1

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return bc_counter, t


# ── Lançamento do Chromium ────────────────────────────────────────────────────

def wait_for_cdp(port: int, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def launch_chromium(seen_ids: set[int]) -> tuple[subprocess.Popen, list[int], Path, int]:
    """Launch Chrome in its own process group; returns (proc, bc_counter, user_data, pgid).

    Starts a daemon thread that writes V8 log lines directly to disk as they arrive,
    so no post-site drain is needed and memory stays bounded regardless of site weight.
    """
    user_data = Path(tempfile.mkdtemp(prefix="chromium_crawl_"))
    flags = [
        str(CHROMIUM_BIN),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={user_data}",
        f"--disable-extensions-except={EXTENSION}",
        f"--load-extension={EXTENSION}",
        "--no-sandbox",
        "--no-first-run",
        "--no-restore-state",
        "--disable-features=ScriptStreaming",
        "--js-flags=--single-threaded",
        "--headless=new",
        "--ozone-platform=x11",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disk-cache-size=0",
        "--media-cache-size=0",
        # UA idêntico ao do Firefox crawler para que ambos recebam o mesmo bundle JS
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "about:blank",
    ]

    proc = subprocess.Popen(flags, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    pgid = os.getpgid(proc.pid)
    bc_counter, _ = _make_stderr_writer(proc, seen_ids)

    print(f"Chrome PID {proc.pid} PGID {pgid} — aguardando CDP na porta {CDP_PORT}...")
    if not wait_for_cdp(CDP_PORT):
        _killpg(pgid, proc)
        shutil.rmtree(user_data, ignore_errors=True)
        raise RuntimeError("Chromium não abriu porta CDP a tempo — verifique o binário")
    print("CDP disponível.")

    return proc, bc_counter, user_data, pgid


# ── Crawling ──────────────────────────────────────────────────────────────────

_TYPING_WORD = "hello"
_TEXT_INPUT_SELECTOR = (
    'input[type="text"], input[type="search"], input[type="email"], '
    'textarea, input:not([type])'
)


def type_in_textbox(page) -> bool:
    try:
        for el in page.query_selector_all(_TEXT_INPUT_SELECTOR):
            if el.is_visible():
                page.evaluate("""([el, word]) => {
                    el.click(); el.focus();
                    var proto = el.tagName === 'TEXTAREA'
                        ? window.HTMLTextAreaElement.prototype
                        : window.HTMLInputElement.prototype;
                    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                    for (var i = 0; i < word.length; i++) {
                        var ch = word[i];
                        el.dispatchEvent(new KeyboardEvent('keydown',  {key: ch, bubbles: true}));
                        el.dispatchEvent(new KeyboardEvent('keypress', {key: ch, bubbles: true}));
                        setter.call(el, el.value + ch);
                        el.dispatchEvent(new InputEvent('input', {data: ch, bubbles: true, inputType: 'insertText'}));
                        el.dispatchEvent(new KeyboardEvent('keyup',    {key: ch, bubbles: true}));
                    }
                }""", [el, _TYPING_WORD])
                return True
    except Exception:
        pass
    return False


def collect_fp_calls(page) -> list[dict]:
    all_calls: list[dict] = []
    for frame in page.frames:
        if not frame.url.startswith("http"):
            continue  # content script can't run in about:blank / data: / blob: frames
        try:
            calls = frame.evaluate(
                "JSON.parse(JSON.stringify(window.__fpCalls || []))"
            )
            all_calls.extend(calls)
        except Exception:
            pass
    return all_calls


def crawl_site(page, url: str) -> tuple[list[dict], bool]:
    t0 = time.time()
    def _ts() -> str:
        return f"t+{time.time()-t0:.1f}s"

    page.set_default_timeout(10_000)  # cap every Playwright op (mouse, evaluate, etc.)
    print(f"  [{_ts()}] goto...")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
    except Exception as e:
        print(f"  [{_ts()}] goto falhou: {e}")
        return [], False

    print(f"  [{_ts()}] página carregada")
    vw = (page.viewport_size or {}).get("width", 1280)
    vh = (page.viewport_size or {}).get("height", 720)

    try:
        page.mouse.move(vw // 2, vh // 2)
    except Exception:
        pass
    time.sleep(0.2)

    try:
        page.evaluate(
            "document.body.dispatchEvent(new MouseEvent('click',"
            "{bubbles:true,cancelable:true,clientX:640,clientY:360}))"
        )
    except Exception:
        pass
    time.sleep(0.2)

    typed = type_in_textbox(page)
    if typed:
        time.sleep(0.2)

    waypoints = [
        (int(vw * 0.25), int(vh * 0.30)),
        (int(vw * 0.75), int(vh * 0.30)),
        (int(vw * 0.50), int(vh * 0.50)),
        (int(vw * 0.25), int(vh * 0.70)),
        (int(vw * 0.75), int(vh * 0.70)),
    ]

    print(f"  [{_ts()}] scroll/dwell...")
    for i in range(DWELL_SECONDS):
        try:
            page.mouse.wheel(0, 300)
        except Exception as e:
            print(f"  [{_ts()}] mouse.wheel falhou no scroll {i+1}: {e}")
            break
        try:
            x, y = waypoints[i % len(waypoints)]
            page.mouse.move(x, y)
        except Exception:
            pass
        time.sleep(1)

    print(f"  [{_ts()}] coletando fp_calls")
    time.sleep(FP_WAIT_EXTRA)
    calls = collect_fp_calls(page)
    print(f"  [{_ts()}] {len(calls)} fp_calls coletados")
    return calls, typed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Crawl a single URL instead of the Tranco list")
    args = parser.parse_args()

    if not CHROMIUM_BIN.exists():
        sys.exit(f"[ERRO] Binário não encontrado: {CHROMIUM_BIN}\n"
                 "Execute o build no PCAD primeiro.")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.url:
        all_sites = [args.url]
        completed = set()
    else:
        all_sites = load_target_sites()
        completed = load_completed()
    pending   = [s for s in all_sites if s not in completed]

    print(f"Tranco top-{CRAWL_LIMIT}: {len(all_sites)} sites carregados")
    print(f"Concluídos: {len(completed)} | Pendentes: {len(pending)}")

    if not pending:
        print("Nada a fazer.")
        return

    # Preload desabilitado — economiza ~700 MB de RAM; dupes entre sessões são
    # deduplicados pelo post_processor.py durante o join.
    seen_ids: set[int] = set()
    print("Bytecodes já no log: (preload desabilitado)")

    # Kill orphaned Chrome processes and clean up temp dirs from previous crashes
    subprocess.run(["pkill", "-9", "-f", str(CHROMIUM_BIN)], capture_output=True)
    time.sleep(0.5)
    for _d in Path(tempfile.gettempdir()).glob("chromium_crawl_*"):
        shutil.rmtree(_d, ignore_errors=True)

    proc, bc_counter, user_data, pgid = launch_chromium(seen_ids)
    proc_ref = [proc]   # mutable refs shared with watchdog threads
    pgid_ref = [pgid]
    total_fp = 0
    site_count = 0
    sites_since_restart = 0

    def _do_restart(reason: str, chrome_dead: bool = False) -> None:
        nonlocal proc, bc_counter, user_data, pgid, sites_since_restart, browser, context
        print(f"\n[RESTART] {reason}")
        if not chrome_dead:
            try:
                browser.disconnect()
            except Exception:
                pass
        _killpg(pgid, proc)
        shutil.rmtree(user_data, ignore_errors=True)
        proc, bc_counter, user_data, pgid = launch_chromium(seen_ids)
        proc_ref[0] = proc
        pgid_ref[0] = pgid
        browser = p.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}", timeout=15000)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        sites_since_restart = 0
        print("[RESTART] Chromium reiniciado.")

    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()

            for url in pending:
                # Recovery: watchdog killed Chrome on the previous site
                if proc.poll() is not None:
                    _do_restart("Chrome morreu (watchdog/crash) — recuperando",
                                chrome_dead=True)

                # Periodic restart to cap process memory growth
                elif sites_since_restart > 0 and sites_since_restart % CHROME_RESTART_EVERY == 0:
                    _do_restart(f"Reiniciando Chromium após {sites_since_restart} sites...")

                print(f"\n→ {url}")
                mark_completed(url)  # before crawl: survives SIGKILL, no retry on stuck sites
                bc_before = bc_counter[0]
                page = context.new_page()
                calls: list[dict] = []
                typed = False
                wd, wd_fired, cancel_nuke = _make_watchdog(SITE_TIMEOUT_S, proc_ref, pgid_ref)
                wd.start()
                try:
                    calls, typed = crawl_site(page, url)
                except Exception as e:
                    print(f"  [ERRO] {e}")
                finally:
                    wd.cancel()
                    if wd_fired.is_set():
                        cancel_nuke()
                        print("  [WATCHDOG] crawl_site retornou após kill — nuke cancelado")
                    elif proc.poll() is not None:
                        print("  [WARN] Chrome morreu (OOM/crash) antes do watchdog — pulando page.close()")
                    else:
                        # Skip page.close() — it hangs on service workers / CDP
                        # timeouts for heavy sites and Playwright's sync API can't
                        # be safely called from a timeout thread (greenlet model).
                        # Accumulated pages are cleaned up by the periodic Chrome
                        # restart every CHROME_RESTART_EVERY sites.
                        pass

                # File writes — outside the watchdog window
                for c in calls:
                    c["page_url"] = url
                with open(FP_CALLS_FILE, "a", encoding="utf-8") as f:
                    for c in calls:
                        json.dump(c, f)
                        f.write("\n")

                # bc_counter is updated by the background writer thread continuously;
                # a brief sleep lets in-flight stderr lines finish before we snapshot.
                time.sleep(0.3)
                saved = bc_counter[0] - bc_before

                total_fp += len(calls)
                typed_tag = "[+] typed" if typed else "[-] no input"
                print(f"  {typed_tag} | {len(calls)} fp_calls | +{saved} bc")

                site_count += 1
                sites_since_restart += 1

            try:
                browser.close()
            except Exception:
                pass

    except Exception as e:
        print(f"[ERRO] Sessão interrompida: {e}")
    finally:
        _killpg(pgid, proc)
        shutil.rmtree(user_data, ignore_errors=True)

    print(f"\nCrawl concluído. FP calls totais: {total_fp} | Bytecodes totais: {bc_counter[0]}")
    print("Próximo passo: python pipeline/post_processor.py")


if __name__ == "__main__":
    main()
