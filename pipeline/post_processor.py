"""
Post-processor: join bytecodes ↔ fp_calls → labeled multilabel dataset.

Memory design (avoids OOM on large bytecodes files):
  - fp_calls: aggregated by DuckDB → compact lookup table (~225K rows, ~50 MB)
  - bytecodes: streamed line-by-line in Python; dedup via FNV-1a hash set
  - per-batch (10K rows) join + heuristics run by DuckDB against the lookup table
  - parquet written incrementally via PyArrow ParquetWriter

Peak Python RSS: ~2 GB (dominated by the dedup hash set at ~37M int entries).
No large temp files needed.

Idempotent: writes to .parquet.tmp, renames to final path only on clean completion.

Usage:
    python post_processor.py                  # both browsers
    python post_processor.py --browser Firefox
    python post_processor.py --browser Chromium
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

LOGS = Path(__file__).resolve().parent.parent / "data"

FIREFOX_BYTECODES  = LOGS / "firefox_sm_bytecodes.jsonl.gz"
FIREFOX_FP_CALLS   = LOGS / "firefox_fp_calls.jsonl"
CHROMIUM_BYTECODES = LOGS / "chromium_v8_bytecodes.jsonl.gz"
CHROMIUM_FP_CALLS  = LOGS / "chromium_fp_calls.jsonl"
OUTPUT_FILE        = LOGS / "labeled_dataset.parquet"
STATS_FILE         = LOGS / "labeled_dataset_stats.json"

BATCH_SIZE    = 10_000  # rows per DuckDB labeling batch
BYTECODES_CAP = 50_000  # truncate at 50K chars (avg is 610; cap affects only outliers)
LOG_EVERY     = 500_000

PARQUET_SCHEMA = pa.schema([
    ("browser",         pa.string()),
    ("injected_id",     pa.string()),
    ("script_url",      pa.string()),
    ("script_id",       pa.int64()),
    ("start_pos",       pa.int64()),
    ("end_pos",         pa.int64()),
    ("callees",         pa.string()),   # comma-separated injected_ids (V8 only)
    ("function_name",   pa.string()),
    ("bytecodes",       pa.string()),
    ("label_canvas",    pa.int8()),
    ("label_audio",     pa.int8()),
    ("label_webrtc",    pa.int8()),
    ("label_font",      pa.int8()),
    ("label_webgl",     pa.int8()),
    ("label_screen",    pa.int8()),
    ("label_navigator", pa.int8()),
    ("is_fp",           pa.bool_()),
])

_ANON = "('anonymous', '(anonymous)', '<anonymous>', 'Object.<anonymous>', '')"

# ── Heuristic SQL fragments — reference alias "f" from the LEFT JOIN ─────────

_H = {
"canvas": """
    list_contains(f.apis,'canvas.toDataURL')
    AND list_contains(f.apis,'canvas.fillText')
    AND COALESCE(f.filltext_maxlen,0)>=10
    AND NOT COALESCE(list_contains(f.apis,'canvas.save'),FALSE)
    AND NOT COALESCE(list_contains(f.apis,'canvas.restore'),FALSE)
    AND NOT COALESCE(list_contains(f.apis,'canvas.addEventListener'),FALSE)
""",
"audio": """
    list_contains(f.apis,'audio.createOscillator')
    AND list_contains(f.apis,'audio.createDynamicsCompressor')
    AND list_contains(f.apis,'audio.startRendering')
    AND list_contains(f.apis,'audio.connect')
    AND (list_contains(f.apis,'audio.getChannelData')
         OR list_contains(f.apis,'audio.copyFromChannel'))
""",
"webrtc": """
    (list_contains(f.apis,'webrtc.createDataChannel')
     OR list_contains(f.apis,'webrtc.createOffer'))
    AND list_contains(f.apis,'webrtc.setLocalDescription')
""",
"font":      "COALESCE(f.measuretext_count,0)>=20 AND COALESCE(f.font_distinct,0)>20",
"webgl":     "COALESCE(f.webgl_fp_count,0)>0",
"screen": """
    (list_contains(f.apis,'screen.colorDepth') OR list_contains(f.apis,'screen.pixelDepth'))
    AND (list_contains(f.apis,'screen.width') OR list_contains(f.apis,'screen.height')
         OR list_contains(f.apis,'screen.availWidth'))
""",
"navigator": "COALESCE(f.nav_distinct,0)>=3",
}

_LABEL_COLS = "\n".join(
    f"    CASE WHEN {sql} THEN 1 ELSE 0 END::TINYINT AS label_{name},"
    for name, sql in _H.items()
)
_IS_FP = " OR\n    ".join(f"({sql})" for sql in _H.values())


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fnv1a(s: str) -> int:
    h = 14695981039346656037
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _bc_key(b: dict) -> int:
    """Unique dedup key per function.

    Named functions: FNV-1a(url + '\\0' + name).
    Anonymous functions (Chromium): FNV-1a(url + '\\0' + start_pos + '\\0' + end_pos)
      because all anonymous functions share the same name string.
    Firefox anonymous functions are already renamed to anon_injected_XXX by
    SpiderMonkey, so they use the named path.
    """
    fn  = b.get("function_name", "")
    url = b.get("script_url", "")
    if fn in (None, "anonymous", "(anonymous)", "<anonymous>", "Object.<anonymous>", ""):
        return _fnv1a(f"{url}\x00{b.get('start_pos','')}\x00{b.get('end_pos','')}")
    return _fnv1a(f"{url}\x00{fn}")


# ── fp_agg: aggregate fp_calls into a compact DuckDB lookup table ─────────────

def _build_fp_agg(conn: duckdb.DuckDBPyConnection,
                  tag: str, fp_file: Path,
                  url_col: str, name_col: str) -> str:
    tbl = f"{tag}_fp_agg"
    conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.execute(f"""
        CREATE TABLE {tbl} AS
        SELECT
            {url_col}::VARCHAR  AS script_url,
            {name_col}::VARCHAR AS func_name,
            ARRAY_AGG(DISTINCT api::VARCHAR) AS apis,
            MAX(CASE WHEN api='canvas.fillText' AND arg IS NOT NULL
                THEN LENGTH(arg::VARCHAR) ELSE 0 END)                   AS filltext_maxlen,
            COUNT(CASE WHEN api='canvas.measureText' THEN 1 END)        AS measuretext_count,
            COUNT(DISTINCT CASE WHEN api='canvas.font'
                THEN COALESCE(arg::VARCHAR,'') END)                     AS font_distinct,
            COUNT(CASE WHEN api='webgl.getParameter'
                AND TRY_CAST(arg AS VARCHAR)
                    IN ('37445','37446','37442','37443') THEN 1 END)     AS webgl_fp_count,
            COUNT(DISTINCT api::VARCHAR)
                FILTER (WHERE api::VARCHAR LIKE 'navigator.%')           AS nav_distinct
        FROM read_ndjson_auto('{fp_file}')
        WHERE {name_col} IS NOT NULL
            AND {name_col}::VARCHAR NOT IN {_ANON}
        GROUP BY {url_col}, {name_col}
    """)
    n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {n:,} pares (url, func) no índice de fp_calls")
    return tbl


# ── Stream bytecodes, dedup in Python, label in DuckDB ───────────────────────

def _process_browser(conn: duckdb.DuckDBPyConnection,
                     browser: str,
                     bc_file: Path,
                     fp_tbl: str,
                     pw: pq.ParquetWriter) -> tuple[int, int]:
    label_q = f"""
        SELECT
            '{browser}'::VARCHAR        AS browser,
            b.injected_id,
            b.script_url,
            TRY_CAST(b.sid AS BIGINT)   AS script_id,
            b.function_name,
            b.bytecodes,
{_LABEL_COLS}
            (({_IS_FP})) AS is_fp
        FROM batch_input b
        LEFT JOIN {fp_tbl} f
            ON b.script_url = f.script_url AND b.function_name = f.func_name
    """

    seen:    set[int] = set()
    bufs = {k: [] for k in
            ["injected_id", "script_url", "sid", "start_pos", "end_pos",
             "callees", "function_name", "bytecodes"]}
    total = fp_total = processed = 0

    def _flush():
        nonlocal total, fp_total
        if not bufs["injected_id"]:
            return
        arrow_in = pa.table({
            "injected_id":   pa.array(bufs["injected_id"],   type=pa.string()),
            "script_url":    pa.array(bufs["script_url"],    type=pa.string()),
            "sid":           pa.array(bufs["sid"],           type=pa.string()),
            "function_name": pa.array(bufs["function_name"], type=pa.string()),
            "bytecodes":     pa.array(bufs["bytecodes"],     type=pa.string()),
        })
        conn.register("batch_input", arrow_in)
        result = conn.execute(label_q).to_arrow_table()
        conn.unregister("batch_input")

        # Append position + callee columns (not used in labeling SQL, added here)
        result = result.append_column(
            "start_pos", pa.array(bufs["start_pos"], type=pa.int64()))
        result = result.append_column(
            "end_pos",   pa.array(bufs["end_pos"],   type=pa.int64()))
        result = result.append_column(
            "callees",   pa.array(bufs["callees"],   type=pa.string()))

        # Reorder to match schema
        result = result.select([f.name for f in PARQUET_SCHEMA])
        pw.write_table(result)
        total    += len(result)
        fp_total += result.column("is_fp").to_pylist().count(True)
        for k in bufs:
            bufs[k].clear()

    opener = gzip.open if bc_file.suffix == ".gz" else open
    with opener(bc_file, "rt", encoding="utf-8") as f:
        try:
          for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = _bc_key(b)
            if key in seen:
                continue
            seen.add(key)

            bufs["injected_id"].append(b.get("injected_id") or "")
            bufs["script_url"].append(b.get("script_url") or "")
            bufs["sid"].append(str(b.get("script_id") or ""))
            bufs["start_pos"].append(int(b.get("start_pos") or 0))
            bufs["end_pos"].append(int(b.get("end_pos") or 0))
            # callees: V8 emits a JSON array; SM doesn't have this field
            raw_callees = b.get("callees", [])
            bufs["callees"].append(",".join(raw_callees) if raw_callees else "")
            bufs["function_name"].append(b.get("function_name") or "")
            bufs["bytecodes"].append((b.get("bytecodes") or "")[:BYTECODES_CAP])
            processed += 1

            if len(bufs["injected_id"]) >= BATCH_SIZE:
                _flush()

            if processed % LOG_EVERY == 0:
                print(f"  [{processed:,} únicas processadas | {fp_total} FP]")
        except Exception as e:
            print(f"  [AVISO] Leitura interrompida após {processed:,} linhas: {e}")

    _flush()
    del seen
    print(f"  Concluído: {processed:,} funções únicas | FP: {fp_total}")
    return total, fp_total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Join bytecodes ↔ fp_calls → parquet")
    parser.add_argument(
        "--browser", choices=["Firefox", "Chromium", "all"], default="all",
        help="Which browser to process (default: all)",
    )
    args = parser.parse_args()

    todo = []
    for browser, bc, fp, uc, nc in [
        ("Firefox",  FIREFOX_BYTECODES,  FIREFOX_FP_CALLS,  "scriptUrl", "funcName"),
        ("Chromium", CHROMIUM_BYTECODES, CHROMIUM_FP_CALLS, "scriptUrl", "funcName"),
    ]:
        if args.browser not in (browser, "all"):
            continue
        if not bc.exists():
            print(f"[SKIP] {bc.name} não encontrado")
            continue
        if not fp.exists():
            print(f"[SKIP] {fp.name} não encontrado")
            continue
        todo.append((browser, bc, fp, uc, nc))

    if not todo:
        sys.exit("[ERRO] Nenhum arquivo de entrada encontrado.")

    # DuckDB in-memory — only fp_agg tables live here (~50 MB each)
    conn = duckdb.connect()

    tmp = OUTPUT_FILE.with_suffix(".parquet.tmp")
    tmp.unlink(missing_ok=True)
    pw = pq.ParquetWriter(tmp, PARQUET_SCHEMA, compression="zstd")

    total_records = total_fp = 0
    try:
        for browser, bc_file, fp_file, url_col, name_col in todo:
            print(f"\n── {browser} ──────────────────────────────────────")
            print(f"  Carregando fp_calls de {fp_file.name}...")
            fp_tbl = _build_fp_agg(conn, browser.lower(), fp_file, url_col, name_col)

            print(f"  Streamindo bytecodes de {bc_file.name}...")
            n, fp = _process_browser(conn, browser, bc_file, fp_tbl, pw)
            total_records += n
            total_fp      += fp
    finally:
        pw.close()

    tmp.rename(OUTPUT_FILE)
    mb = OUTPUT_FILE.stat().st_size / 1_048_576
    print(f"\nDataset: {total_records:,} funções, {total_fp:,} FP → {OUTPUT_FILE.name} ({mb:.1f} MB)")

    # ── Containment propagation ───────────────────────────────────────────────
    # For each non-FP function whose source range directly contains at least one
    # FP-labeled function (by start_pos/end_pos nesting), mark it FP.
    # This labels orchestrator/wrapper functions that call FP leaf functions but
    # whose own bytecodes are generic. Writes labeled_dataset_propagated.parquet.
    print("\nRunning containment propagation...")
    PROP_FILE = OUTPUT_FILE.with_name("labeled_dataset_propagated.parquet")
    prop_tmp  = PROP_FILE.with_suffix(".parquet.tmp")
    prop_tmp.unlink(missing_ok=True)

    # Load only the columns needed for propagation into DuckDB (avoids full
    # 5 GB load — ~400 MB for these 4 columns across 80M rows)
    conn.execute(f"""
        CREATE TABLE base AS
        SELECT
            injected_id,
            script_url,
            start_pos,
            end_pos,
            is_fp
        FROM read_parquet('{OUTPUT_FILE}')
        WHERE start_pos > 0 AND end_pos > start_pos
    """)

    # For each function F, check whether any other function G in the same script
    # satisfies: G.start_pos > F.start_pos AND G.end_pos < F.end_pos AND G.is_fp.
    # These are functions directly nested inside F that are already FP.
    conn.execute("""
        CREATE TABLE fp_parents AS
        SELECT DISTINCT f.injected_id AS injected_id
        FROM base f
        JOIN base g
          ON  g.script_url = f.script_url
          AND g.start_pos  > f.start_pos
          AND g.end_pos    < f.end_pos
          AND g.is_fp      = TRUE
          AND f.is_fp      = FALSE
    """)

    n_new = conn.execute("SELECT COUNT(*) FROM fp_parents").fetchone()[0]
    print(f"  {n_new:,} non-FP functions promoted to FP by containment")

    # Rewrite the parquet with is_fp updated for promoted functions
    conn.execute(f"""
        COPY (
            SELECT
                b.browser, b.injected_id, b.script_url, b.script_id,
                b.start_pos, b.end_pos, b.callees, b.function_name, b.bytecodes,
                b.label_canvas, b.label_audio, b.label_webrtc, b.label_font,
                b.label_webgl, b.label_screen, b.label_navigator,
                CASE WHEN b.is_fp OR p.injected_id IS NOT NULL
                     THEN TRUE ELSE FALSE END AS is_fp
            FROM read_parquet('{OUTPUT_FILE}') b
            LEFT JOIN fp_parents p USING (injected_id)
        ) TO '{prop_tmp}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    prop_tmp.rename(PROP_FILE)
    prop_mb = PROP_FILE.stat().st_size / 1_048_576
    total_fp_prop = total_fp + n_new
    print(f"  Propagated dataset: {total_records:,} funções, {total_fp_prop:,} FP → "
          f"{PROP_FILE.name} ({prop_mb:.1f} MB)")

    # Query label breakdown and write stats sidecar (read by status.py without deps)
    label_cols = ["canvas", "audio", "webrtc", "font", "webgl", "screen", "navigator"]
    sel = ", ".join(
        f"COUNT(*) FILTER (WHERE label_{c}) AS {c}" for c in label_cols
    )
    row = conn.execute(
        f"SELECT {sel} FROM read_parquet('{OUTPUT_FILE}')"
    ).fetchone()
    labels = dict(zip(label_cols, row))
    stats = {
        "total":           total_records,
        "fp":              total_fp,
        "fp_propagated":   total_fp_prop,
        "new_fp_from_containment": n_new,
        "size_mb":         round(mb, 1),
        "generated":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labels":          labels,
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
