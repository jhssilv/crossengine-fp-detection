"""
Firefox crawler capturing SpiderMonkey bytecodes and fingerprinting API calls.

Idempotent: finished URLs are recorded in DONE_FILE. Re-running the script
resumes from the first pending URL without revisiting finished sites.

Flow per batch of BATCH_SIZE sites:
  1. Launch the instrumented Firefox (geckodriver → geckodriver_raw.log)
  2. Load the shared extension (MAIN world content script)
  3. Crawl BATCH_SIZE sites; after each site: append fp_calls + mark done
  4. Close Firefox; parse the batch geckodriver log → append bytecodes
  5. Repeat until no sites are pending
"""
import argparse
import gzip
import json
import os
import threading
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ── Paths ─────────────────────────────────────────────────────────────────────
TCC_ROOT    = Path(__file__).resolve().parent.parent
FIREFOX_BIN = TCC_ROOT / "navegadores" / "spidermonkey-tcc" / "gecko-dev" / \
              "obj-x86_64-pc-linux-gnu" / "dist" / "bin" / "firefox"
EXTENSION   = TCC_ROOT / "extension"
LOGS_DIR    = TCC_ROOT / "data"
TRANCO_FILE = TCC_ROOT / "crawlers" / "tranco_top10k.csv"

GECKODRIVER_LOG = LOGS_DIR / "geckodriver_raw.log"
SM_CLEAN_LOG    = LOGS_DIR / "firefox_sm_bytecodes.jsonl.gz"
FP_CALLS_FILE   = LOGS_DIR / "firefox_fp_calls.jsonl"
DONE_FILE       = LOGS_DIR / "firefox_done.txt"

# ── Config ────────────────────────────────────────────────────────────────────
CRAWL_LIMIT   = 10_000  # number of Tranco sites to crawl
BATCH_SIZE    = 30      # restart Firefox every N sites (bounds geckodriver log size)
DWELL_SECONDS = 4       # seconds of scrolling per page
FP_WAIT_EXTRA = 2       # extra wait for lazy fingerprinting
PAGE_TIMEOUT    = 15    # page load timeout (seconds)
SITE_TIMEOUT_FF = 60    # watchdog: kill Firefox if a site hangs (seconds)

CHROME_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

INTERNAL_PREFIXES = (
    "resource://", "chrome://", "about:", "moz-src:", "file:",
    "moz-extension://", "moz-nullprincipal:", "<anonymous code>", "data:",
)


# ── Watchdog ─────────────────────────────────────────────────────────────────

def _make_watchdog_ff(timeout: float, driver: webdriver.Firefox):
    """Kill geckodriver after timeout; nuke Python 15s later if still blocked.

    Returns (timer, fired_event, cancel_nuke).
    """
    fired    = threading.Event()
    nuke_ref = [None]

    def _kill():
        fired.set()
        print(f"\n  [WATCHDOG] Firefox stuck >{timeout:.0f}s, killing geckodriver")
        try:
            driver.service.process.kill()
        except Exception as e:
            print(f"  [WATCHDOG] kill failed: {e}")
        print("  [WATCHDOG] geckodriver killed. Nuclear timer in 15s in case Python hangs.")

        def _self_kill():
            print("  [WATCHDOG NUCLEAR] Python stuck after kill, restarting process")
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


# ── SpiderMonkey parser (append, cross-batch dedup) ───────────────────────────

def _sm_key(url: str, name: str) -> int:
    """FNV-1a 64-bit hash of url+name — replaces string tuples in seen set (~10x smaller)."""
    h = 14695981039346656037
    for c in (url + "\x00" + name).encode():
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def parse_spidermonkey_batch(seen: set[int]) -> int:
    """Parse the current batch geckodriver log and append new bytecodes.

    seen: set of (script_url, func_name) already written in earlier sessions.
    Updated in place to deduplicate across batches of the same session.
    """
    if not GECKODRIVER_LOG.exists():
        return 0

    count = skipped = dupes = 0
    with open(GECKODRIVER_LOG, "r", encoding="utf-8", errors="replace") as f_in, \
         gzip.open(SM_CLEAN_LOG, "at", encoding="utf-8") as f_out:
        it = iter(f_in)
        for line in it:
            if line.strip() != "[SpiderMonkey Log]":
                continue
            try:
                script_url = next(it).split("Script URL: ", 1)[1].strip()
                script_id  = next(it).split("Internal Index: ", 1)[1].strip()
                start_pos  = int(next(it).split("Start: ", 1)[1].strip())
                end_pos    = int(next(it).split("End: ", 1)[1].strip())
                func_name  = next(it).split("Function name: ", 1)[1].strip()
                bytecodes  = next(it).split("Bytecode: [", 1)[1].rstrip().rstrip("]").strip()
            except StopIteration:
                break
            except (IndexError, ValueError):
                continue
            if any(script_url.startswith(p) for p in INTERNAL_PREFIXES):
                skipped += 1
            else:
                key = _sm_key(script_url, func_name)
                if key in seen:
                    dupes += 1
                else:
                    seen.add(key)
                    record = {
                        "browser":       "Firefox",
                        "injected_id":   func_name,
                        "script_url":    script_url,
                        "script_id":     script_id,
                        "start_pos":     start_pos,
                        "end_pos":       end_pos,
                        "function_name": func_name,
                        "bytecodes":     bytecodes,
                    }
                    json.dump(record, f_out)
                    f_out.write("\n")
                    count += 1

    print(f"  SM: +{count} functions ({skipped} internal, {dupes} dupes filtered)")
    return count


# ── Crawl ─────────────────────────────────────────────────────────────────────

def start_firefox() -> webdriver.Firefox:
    options = Options()
    options.binary_location = str(FIREFOX_BIN)
    options.page_load_strategy = "none"
    options.add_argument("--headless")
    options.set_preference("general.useragent.override", CHROME_UA)

    service = Service(
        executable_path="/usr/local/bin/geckodriver",
        log_output=str(GECKODRIVER_LOG),
    )
    driver = webdriver.Firefox(service=service, options=options)
    driver.set_page_load_timeout(PAGE_TIMEOUT)

    try:
        driver.install_addon(str(EXTENSION), temporary=True)
    except Exception as e:
        print(f"  [WARN] Extension not loaded: {e}")
    return driver


_WAYPOINTS = [
    (300, 250), (900, 250), (640, 400), (300, 550), (900, 550),
]

_TYPING_WORD = "hello"
_TEXT_INPUT_SELECTORS = (
    'input[type="text"]',
    'input[type="search"]',
    'input[type="email"]',
    'textarea',
)


_JS_TYPE = """
(function(el, word) {
    el.click();
    el.focus();
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
})(arguments[0], arguments[1]);
"""


def type_in_textbox(driver: webdriver.Firefox) -> bool:
    combined = ", ".join(_TEXT_INPUT_SELECTORS)
    try:
        el = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, combined))
        )
        driver.execute_script(_JS_TYPE, el, _TYPING_WORD)
        return True
    except (NoSuchElementException, TimeoutException):
        return False  # no input on this page — expected
    except Exception as e:
        print(f"  [WARN] type_in_textbox: {e}")
        return False


def crawl_site(driver: webdriver.Firefox, url: str) -> tuple[list[dict], bool]:
    t0 = time.time()
    def _ts() -> str:
        return f"t+{time.time()-t0:.1f}s"

    try:
        driver.get(url)
    except Exception as e:
        print(f"  [WARN] get() failed ({_ts()}): {e}")
        return [], False

    print(f"  [{_ts()}] page loaded")

    try:
        driver.execute_script(
            "document.dispatchEvent(new MouseEvent('mousemove',"
            "{bubbles:true,clientX:640,clientY:360}));"
        )
    except Exception:
        pass
    time.sleep(0.2)

    try:
        driver.execute_script(
            "document.body.dispatchEvent(new MouseEvent('click',"
            "{bubbles:true,cancelable:true,clientX:640,clientY:360}));"
        )
    except Exception:
        pass
    time.sleep(0.2)

    typed = type_in_textbox(driver)
    if typed:
        time.sleep(0.2)

    print(f"  [{_ts()}] iniciando scroll ({DWELL_SECONDS}s)")
    for i, _ in enumerate(range(DWELL_SECONDS)):
        try:
            driver.execute_script("window.scrollBy(0, 300);")
        except Exception as e:
            print(f"  [{_ts()}] scrollBy failed at step {i+1}: {e}")
            break
        try:
            x, y = _WAYPOINTS[i % len(_WAYPOINTS)]
            driver.execute_script(
                f"document.dispatchEvent(new MouseEvent('mousemove',"
                f"{{bubbles:true,clientX:{x},clientY:{y}}}));"
            )
        except Exception:
            pass
        time.sleep(1)

    print(f"  [{_ts()}] coletando fp_calls")
    time.sleep(FP_WAIT_EXTRA)

    try:
        calls = driver.execute_script(
            "return JSON.parse(JSON.stringify(window.__fpCalls || []))"
        )
        print(f"  [{_ts()}] {len(calls or [])} fp_calls coletados")
        return calls or [], typed
    except Exception as e:
        print(f"  [WARN] fp_calls failed ({_ts()}): {e}")
        return [], typed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Crawl a single URL instead of the Tranco list")
    args = parser.parse_args()

    if not FIREFOX_BIN.exists():
        raise FileNotFoundError(f"Firefox not found: {FIREFOX_BIN}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.url:
        all_sites = [args.url]
        completed = set()
    else:
        all_sites = load_target_sites()
        completed = load_completed()
    pending   = [s for s in all_sites if s not in completed]

    print(f"Tranco top-{CRAWL_LIMIT}: {len(all_sites)} sites loaded")
    print(f"Done: {len(completed)} | Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Preload disabled (saves ~4 GB of RAM); cross-session duplicates are
    # removed by post_processor.py during the join.
    seen_sm: set[int] = set()
    print("Bytecodes already in log: (preload disabled)")

    total_fp = 0
    batch_num = 0

    while pending:
        batch = pending[:BATCH_SIZE]
        pending = pending[BATCH_SIZE:]
        batch_num += 1

        print(f"\n{'='*60}")
        print(f"Batch {batch_num} | {len(batch)} sites | {len(pending)} remaining")

        GECKODRIVER_LOG.write_bytes(b"")  # truncate before each batch
        driver = start_firefox()
        batch_fp = 0

        try:
            for url in batch:
                print(f"\n→ {url}")
                mark_completed(url)  # before crawling: survives SIGKILL, no retry on hanging sites

                wd, wd_fired, cancel_nuke = _make_watchdog_ff(SITE_TIMEOUT_FF, driver)
                wd.start()
                calls: list[dict] = []
                typed = False
                try:
                    calls, typed = crawl_site(driver, url)
                except Exception as e:
                    print(f"  [ERROR] crawl_site: {e}")
                finally:
                    wd.cancel()
                    if wd_fired.is_set():
                        cancel_nuke()
                        print("  [WATCHDOG] crawl_site returned, nuke cancelled")

                if wd_fired.is_set():
                    print("  Firefox died on this site, ending batch early")
                    break

                for c in calls:
                    c["page_url"] = url

                with open(FP_CALLS_FILE, "a", encoding="utf-8") as f:
                    for c in calls:
                        json.dump(c, f)
                        f.write("\n")

                batch_fp += len(calls)
                typed_tag = "[+] typed" if typed else "[-] no input"
                print(f"  {typed_tag} | {len(calls)} fp_calls")

        except Exception as e:
            print(f"[ERROR] Batch interrupted: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        # Parse the batch log (Firefox closed → log is stable)
        new_bc = parse_spidermonkey_batch(seen_sm)
        total_fp += batch_fp
        print(f"Batch {batch_num} done: {batch_fp} fp_calls, {new_bc} new bytecodes")

    print(f"\nCrawl finished. Total FP calls: {total_fp}")
    print("Next step: python pipeline/post_processor.py")


if __name__ == "__main__":
    main()
