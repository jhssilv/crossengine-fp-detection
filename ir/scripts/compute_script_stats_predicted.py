"""
Two-pass script statistics using model predictions (no ground-truth labels).

Streams all rows of translated_dataset.parquet through the existing
LightGBM density model and saves, per script_url:

    density        — mean predicted FP probability
    pred_fp_count  — count of functions with predicted p > 0.5
    size           — total function count in script

Output: ir/script_stats_predicted.json

This is the clean, deployable alternative to compute_script_stats.py
(which uses ground-truth is_fp labels and leaks test-set information).

Usage:
    python ir/scripts/compute_script_stats_predicted.py
"""

import json
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_IR_ROOT     = Path(__file__).parent.parent
MODEL_DIR    = _IR_ROOT / "lgbm_model_density"
DATASET      = _IR_ROOT / "translated_dataset.parquet"
DENSITY_FILE = _IR_ROOT / "script_density.json"   # existing mean-prob map (bootstrap)
OUT          = _IR_ROOT / "script_stats_predicted.json"

BATCH_SIZE = 4096


def main():
    print("Loading density model...")
    with open(MODEL_DIR / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    print("Loading existing density map (bootstrap for density feature)...")
    with open(DENSITY_FILE) as f:
        density_map = json.load(f)
    print(f"  {len(density_map):,} scripts")

    script_sum   = defaultdict(float)   # sum of predicted probabilities
    script_above = defaultdict(int)     # count where p > 0.5
    script_total = defaultdict(int)     # total function count

    pf         = pq.ParquetFile(DATASET)
    n_rg       = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    token_buf  = []
    script_buf = []
    rows_done  = 0
    t0         = time.time()

    def flush():
        if not token_buf:
            return
        tf_mat = tfidf.transform(token_buf)
        # Append the per-function density (from existing map) as the last feature
        densities = np.array(
            [density_map.get(u, 0.0) for u in script_buf], dtype=np.float32
        ).reshape(-1, 1)
        X = hstack([tf_mat, csr_matrix(densities)])
        probs = model.predict_proba(X)[:, 1]
        for url, p in zip(script_buf, probs):
            script_sum[url]   += float(p)
            script_total[url] += 1
            if p > 0.5:
                script_above[url] += 1
        token_buf.clear()
        script_buf.clear()

    print(f"Streaming {total_rows:,} rows across {n_rg} row groups...")
    for rg_idx in range(n_rg):
        if rg_idx % 50 == 0:
            elapsed = time.time() - t0
            pct = 100 * rows_done / total_rows if rows_done else 0
            eta = (elapsed / rows_done) * (total_rows - rows_done) if rows_done else 0
            print(f"  RG {rg_idx}/{n_rg}  {rows_done:,}/{total_rows:,} "
                  f"({pct:.1f}%)  scripts={len(script_total):,}  "
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
    print(f"Unique scripts: {len(script_total):,}")

    stats = {
        url: {
            "density":       script_sum[url] / script_total[url],
            "pred_fp_count": script_above[url],
            "size":          script_total[url],
        }
        for url in script_total
    }

    likely_fp = sum(1 for s in stats.values() if s["density"] >= 0.5)
    any_above = sum(1 for s in stats.values() if s["pred_fp_count"] >= 1)
    print(f"Scripts with mean density >= 0.5: {likely_fp:,}")
    print(f"Scripts with pred_fp_count  >= 1: {any_above:,}")

    with open(OUT, "w") as f:
        json.dump(stats, f)
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
