"""
Post-training evaluation for the LightGBM bigram binary classifier.

Produces two artifacts saved to the model directory:
  inference_benchmark.json  — latency/throughput numbers (mean, p50, p95, p99 ms/function)
  feature_importance.json   — top-N IR bigrams by split-gain importance

The test split is reconstructed deterministically using the same seed and
collect_sample logic as train_lgbm.py, so these numbers match the training run.

Usage:
    python ir/scripts/eval_lgbm.py [--model-dir ir/lgbm_model] [--top-n 30]
"""

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

_IR_ROOT = Path(__file__).parent.parent
DATASET = str(_IR_ROOT / "translated_dataset.parquet")

SEED = 42
NEG_RATIO = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(_IR_ROOT / "lgbm_model"))
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--batch-sizes", default="1,10,100,1000",
                   help="Comma-separated batch sizes for latency sweep")
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed):
    """Identical to train_lgbm.py — must stay in sync."""
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("Estimating FP count (1-in-50 sample)...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled += len(col)
    est_fp = int(fp_sample * total_rows / sampled)
    est_neg = total_rows - est_fp
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(est_neg, 1))
    print(f"  est FP={est_fp:,}  target neg={target_neg:,}  p={sample_prob:.4f}")

    fp_rows, neg_rows = [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 200 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["ir_tokens", "is_fp"]).to_pandas()
        df["is_fp"] = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")
        fp_df = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]
        if len(fp_df):
            fp_rows.extend(fp_df["ir_tokens"].tolist())
        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                neg_rows.extend(neg_df["ir_tokens"].values[keep].tolist())

    X = fp_rows + neg_rows
    Y = np.array([1] * len(fp_rows) + [0] * len(neg_rows), dtype=np.int8)
    print(f"Collected {len(X):,} rows: {len(fp_rows):,} FP + {len(neg_rows):,} non-FP")
    return X, Y


def benchmark_inference(tfidf, model, X_te, batch_sizes, threshold):
    """Measure end-to-end latency (TF-IDF transform + predict_proba) per batch size."""
    results = {}
    for bs in batch_sizes:
        if bs > len(X_te):
            continue
        # Draw the same slice repeatedly for stable timing
        subset = X_te[:bs]
        times_ms = []
        # Warm-up
        _ = model.predict_proba(tfidf.transform(subset))
        for _ in range(max(5, 200 // bs)):
            t0 = time.perf_counter()
            vec = tfidf.transform(subset)
            _ = model.predict_proba(vec)[:, 1] >= threshold
            times_ms.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times_ms)
        results[bs] = {
            "n_reps":         len(times_ms),
            "mean_ms":        round(float(arr.mean()), 4),
            "p50_ms":         round(float(np.percentile(arr, 50)), 4),
            "p95_ms":         round(float(np.percentile(arr, 95)), 4),
            "p99_ms":         round(float(np.percentile(arr, 99)), 4),
            "per_fn_mean_us": round(float(arr.mean() / bs * 1000), 4),
        }
        print(
            f"  batch={bs:>5}  mean={arr.mean():.2f}ms  "
            f"p50={np.percentile(arr,50):.2f}ms  "
            f"p95={np.percentile(arr,95):.2f}ms  "
            f"per-fn={arr.mean()/bs*1000:.2f}µs"
        )
    return results


def feature_importance(tfidf, model, top_n):
    names = np.array(tfidf.get_feature_names_out())
    gains = model.booster_.feature_importance(importance_type="gain")
    splits = model.booster_.feature_importance(importance_type="split")
    order = np.argsort(gains)[::-1][:top_n]
    entries = []
    for rank, idx in enumerate(order, 1):
        entries.append({
            "rank":    rank,
            "feature": names[idx],
            "gain":    round(float(gains[idx]), 2),
            "splits":  int(splits[idx]),
        })
        print(f"  {rank:>3}. {names[idx]:<40}  gain={gains[idx]:>12.1f}  splits={splits[idx]:>6}")
    return entries


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"Loading model from {model_dir} ...")
    with open(model_dir / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(model_dir / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "metrics.json") as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    print(f"  threshold={threshold}  features={meta['n_features']:,}  ngram_max={meta['ngram_max']}")

    print("\nReconstructing test split...")
    pf = pq.ParquetFile(DATASET)
    X, Y = collect_sample(pf, NEG_RATIO, SEED)
    _, X_te, _, _ = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    print(f"Test set: {len(X_te):,} samples")

    print("\n--- Inference benchmark ---")
    bench = benchmark_inference(tfidf, model, X_te, batch_sizes, threshold)

    print("\n--- Feature importance (top {}) ---".format(args.top_n))
    fi = feature_importance(tfidf, model, args.top_n)

    out_bench = model_dir / "inference_benchmark.json"
    out_fi    = model_dir / "feature_importance.json"

    with open(out_bench, "w") as f:
        json.dump(bench, f, indent=2)
    with open(out_fi, "w") as f:
        json.dump(fi, f, indent=2)

    print(f"\nSaved:\n  {out_bench}\n  {out_fi}")


if __name__ == "__main__":
    main()
