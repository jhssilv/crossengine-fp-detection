"""
Step 1 of the full-dataset script density pipeline.

Streams ALL 80M rows of translated_dataset.parquet through the existing
LightGBM model to compute, for each script_url, the mean predicted FP
probability across ALL its functions.

Saves ir/script_density.json: {script_url -> mean_p_fp}

Usage:
    python ir/scripts/compute_script_density.py
"""

import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_IR_ROOT  = Path(__file__).parent.parent
MODEL_DIR = _IR_ROOT / "lgbm_model"
DATASET   = _IR_ROOT / "translated_dataset.parquet"
OUT       = _IR_ROOT / "script_density.json"

BATCH_SIZE = 4096   # functions per TF-IDF transform call


def main():
    print("Loading model...")
    with open(MODEL_DIR / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    script_sum   = defaultdict(float)
    script_count = defaultdict(int)

    pf   = pq.ParquetFile(DATASET)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    token_buf  = []
    script_buf = []
    rows_done  = 0
    t0 = time.time()

    def flush():
        if not token_buf:
            return
        vec   = tfidf.transform(token_buf)
        probs = model.predict_proba(vec)[:, 1]
        for url, p in zip(script_buf, probs):
            script_sum[url]   += float(p)
            script_count[url] += 1
        token_buf.clear()
        script_buf.clear()

    print(f"Streaming {total_rows:,} rows across {n_rg} row groups...")
    for rg_idx in range(n_rg):
        if rg_idx % 50 == 0:
            elapsed = time.time() - t0
            pct = 100 * rows_done / total_rows if rows_done else 0
            eta = (elapsed / rows_done) * (total_rows - rows_done) if rows_done else 0
            print(f"  RG {rg_idx}/{n_rg}  {rows_done:,}/{total_rows:,} "
                  f"({pct:.1f}%)  scripts={len(script_count):,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        df = pf.read_row_group(rg_idx, columns=["ir_tokens", "script_url"]).to_pandas()
        df["ir_tokens"]  = df["ir_tokens"].fillna("")
        df["script_url"] = df["script_url"].fillna("")

        for tok, url in zip(df["ir_tokens"].tolist(), df["script_url"].tolist()):
            token_buf.append(tok)
            script_buf.append(url)
            if len(token_buf) >= BATCH_SIZE:
                flush()

        rows_done += len(df)

    flush()

    elapsed = time.time() - t0
    print(f"\nDone: {rows_done:,} rows in {elapsed:.0f}s ({rows_done/elapsed:.0f} rows/s)")
    print(f"Unique scripts: {len(script_count):,}")

    density = {url: script_sum[url] / script_count[url] for url in script_count}

    fp_scripts  = sum(1 for v in density.values() if v >= 0.5)
    fp_any      = sum(1 for v in density.values() if v >= 0.1)
    print(f"Scripts with mean p_fp >= 0.5 (likely FP): {fp_scripts:,}")
    print(f"Scripts with mean p_fp >= 0.1             : {fp_any:,}")

    with open(OUT, "w") as f:
        json.dump(density, f)
    print(f"Saved → {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
