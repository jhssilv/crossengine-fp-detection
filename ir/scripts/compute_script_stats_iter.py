"""
Iterative script-stats computation for multi-pass self-improvement.

Streams all 68M rows through a given model, accumulates per-script:
    density        — mean predicted FP probability
    fp_count_30    — count where p > 0.30  (soft FP signal)
    fp_count_50    — count where p > 0.50  (moderate FP signal)
    fp_count_70    — count where p > 0.70  (strong FP signal)
    size           — total function count

The context feature vector appended to TF-IDF is auto-detected from the
model's n_features_in_ minus the TF-IDF vocabulary size:
    n_context == 1: [density]                                (v1 density model)
    n_context == 3: [density, log1p(fp_count_50), log1p(size)]  (v2)
    n_context == 5: [density, log1p(fp_count_30), log1p(fp_count_50),
                     log1p(fp_count_70), log1p(size)]            (v3+)

Usage:
    # iteration 2: use v2 model, bootstrap from v1 stats
    python ir/scripts/compute_script_stats_iter.py \\
        --model-dir  ir/lgbm_model_density_v2 \\
        --prev-stats ir/script_stats_predicted.json \\
        --out        ir/script_stats_iter2.json

    # iteration 3: use v3b model, bootstrap from iter2 stats
    python ir/scripts/compute_script_stats_iter.py \\
        --model-dir  ir/lgbm_model_density_v3b \\
        --prev-stats ir/script_stats_iter2.json \\
        --out        ir/script_stats_iter3.json
"""

import argparse
import json
import math
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_IR_ROOT   = Path(__file__).parent.parent
DATASET    = _IR_ROOT / "translated_dataset.parquet"
BATCH_SIZE = 4096


def build_ctx(url, prev, n_context):
    """Build context feature vector for a single URL, matching model's expected n_context."""
    s = prev.get(url, {})
    d   = s.get("density",     0.0)
    f50 = s.get("fp_count_50", s.get("pred_fp_count", 0))
    sz  = s.get("size",        0)
    f30 = s.get("fp_count_30", f50)   # fall back to f50 if not present
    f70 = s.get("fp_count_70", 0)
    if n_context == 1:
        return [d]
    elif n_context == 3:
        return [d, math.log1p(f50), math.log1p(sz)]
    else:   # 5 or any other value — v3+ features
        return [d, math.log1p(f30), math.log1p(f50), math.log1p(f70), math.log1p(sz)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  default=str(_IR_ROOT / "lgbm_model_density_v2"))
    p.add_argument("--prev-stats", default=str(_IR_ROOT / "script_stats_predicted.json"),
                   help="JSON from previous iteration (bootstrap context features)")
    p.add_argument("--out",        default=str(_IR_ROOT / "script_stats_iter.json"))
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    print(f"Loading model from {model_dir}...")
    with open(model_dir / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    n_tfidf   = len(tfidf.vocabulary_)
    n_context = model.n_features_in_ - n_tfidf
    print(f"  TF-IDF vocab: {n_tfidf:,}  context features: {n_context}")

    print(f"Loading bootstrap stats from {args.prev_stats}...")
    with open(args.prev_stats) as f:
        prev = json.load(f)
    print(f"  {len(prev):,} scripts")

    script_sum = defaultdict(float)
    fp_cnt30   = defaultdict(int)
    fp_cnt50   = defaultdict(int)
    fp_cnt70   = defaultdict(int)
    script_tot = defaultdict(int)

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
        ctx    = np.array([build_ctx(u, prev, n_context) for u in script_buf],
                          dtype=np.float32)
        X      = hstack([tf_mat, csr_matrix(ctx)])
        probs  = model.predict_proba(X)[:, 1]
        for url, p in zip(script_buf, probs):
            script_sum[url] += float(p)
            script_tot[url] += 1
            if p > 0.3: fp_cnt30[url] += 1
            if p > 0.5: fp_cnt50[url] += 1
            if p > 0.7: fp_cnt70[url] += 1
        token_buf.clear()
        script_buf.clear()

    print(f"Streaming {total_rows:,} rows across {n_rg} row groups...")
    for rg_idx in range(n_rg):
        if rg_idx % 50 == 0:
            elapsed = time.time() - t0
            pct = 100 * rows_done / total_rows if rows_done else 0
            eta = (elapsed / rows_done) * (total_rows - rows_done) if rows_done else 0
            print(f"  RG {rg_idx}/{n_rg}  {rows_done:,}/{total_rows:,} ({pct:.1f}%)"
                  f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

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
    print(f"Unique scripts: {len(script_tot):,}")

    stats = {
        url: {
            "density":    script_sum[url] / script_tot[url],
            "fp_count_30": fp_cnt30[url],
            "fp_count_50": fp_cnt50[url],
            "fp_count_70": fp_cnt70[url],
            "size":        script_tot[url],
        }
        for url in script_tot
    }

    likely_fp = sum(1 for s in stats.values() if s["density"] >= 0.5)
    print(f"Scripts with mean density >= 0.5: {likely_fp:,}")

    out = Path(args.out)
    with open(out, "w") as f:
        json.dump(stats, f)
    print(f"Saved → {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
