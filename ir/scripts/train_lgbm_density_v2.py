"""
LightGBM binary classifier v2 — clean two-pass script context.

Baseline: lgbm_model_density (bigrams + mean_pred_density) → F1=0.853
Goal:     replace the single density float with 3 deployable features
          derived from model predictions (no ground-truth labels):

    script_density      — mean predicted FP probability per script
    script_pred_fp_log  — log1p(count of functions with p > 0.5 in script)
    script_size_log     — log1p(total function count in script)

All three are computed by compute_script_stats_predicted.py (first pass)
and stored in ir/script_stats_predicted.json.

Usage:
    # 1. Run first pass (once, ~same time as compute_script_density.py)
    python ir/scripts/compute_script_stats_predicted.py

    # 2. Train v2 model
    python ir/scripts/train_lgbm_density_v2.py [--out ir/lgbm_model_density_v2]
"""

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT   = Path(__file__).parent.parent
DATASET    = str(_IR_ROOT / "translated_dataset.parquet")
STATS_FILE = _IR_ROOT / "script_stats_predicted.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio",    type=float, default=10.0)
    p.add_argument("--ngram-max",    type=int,   default=2)
    p.add_argument("--n-estimators", type=int,   default=500)
    p.add_argument("--out",          default=str(_IR_ROOT / "lgbm_model_density_v2"))
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def context_features(url: str, stats: dict) -> np.ndarray:
    s = stats.get(url, {})
    density       = s.get("density",       0.0)
    pred_fp_count = s.get("pred_fp_count", 0)
    size          = s.get("size",          0)
    return np.array([
        density,
        math.log1p(pred_fp_count),
        math.log1p(size),
    ], dtype=np.float32)


def collect_sample(pf, neg_ratio, seed, stats):
    rng        = random.Random(seed)
    n_rg       = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("Estimating FP count from 1-in-50 row group sample...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled   += len(col)
    est_fp      = int(fp_sample * total_rows / sampled)
    target_neg  = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    tokens_list, ctx_list, labels_list = [], [], []
    t0 = time.time()

    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)

        df = pf.read_row_group(
            i, columns=["ir_tokens", "is_fp", "script_url"]
        ).to_pandas()
        df["is_fp"]      = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"]  = df["ir_tokens"].fillna("")
        df["script_url"] = df["script_url"].fillna("")

        fp_mask  = df["is_fp"].values
        neg_mask = ~fp_mask

        for mask, label in [(fp_mask, 1), (neg_mask, 0)]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            if label == 0:
                keep = np.array([rng.random() < sample_prob for _ in range(len(sub))])
                if not keep.any():
                    continue
                sub = sub.iloc[keep]

            toks = sub["ir_tokens"].tolist()
            urls = sub["script_url"].tolist()

            tokens_list.extend(toks)
            ctx_list.extend([context_features(u, stats) for u in urls])
            labels_list.extend([label] * len(toks))

    C = np.array(ctx_list,    dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int8)
    print(f"Collected {len(tokens_list):,}: {Y.sum():,} FP + {(Y==0).sum():,} non-FP")
    return tokens_list, C, Y


def tune_threshold(model, X_vec, Y, n_steps=19):
    probs = model.predict_proba(X_vec)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(Y, probs >= 0.5, zero_division=0)
    print(f"  Threshold: @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1, probs


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not STATS_FILE.exists():
        print(f"ERROR: {STATS_FILE} not found.")
        print("Run: python ir/scripts/compute_script_stats_predicted.py")
        return

    print("Loading script stats (predicted)...")
    with open(STATS_FILE) as f:
        stats = json.load(f)
    print(f"  {len(stats):,} scripts")

    pf = pq.ParquetFile(DATASET)
    tokens, C, Y = collect_sample(pf, args.neg_ratio, args.seed, stats)

    X_tr_raw, X_te_raw, C_tr, C_te, Y_tr, Y_te = train_test_split(
        tokens, C, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr_raw):,}  Test: {len(X_te_raw):,}")
    print(f"Train FP: {Y_tr.sum():,}  Test FP: {Y_te.sum():,}")

    print(f"\nFitting TF-IDF (ngram 1-{args.ngram_max})...")
    tfidf   = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, args.ngram_max))
    X_tr_tf = tfidf.fit_transform(X_tr_raw)
    X_te_tf = tfidf.transform(X_te_raw)
    print(f"  TF-IDF features: {X_tr_tf.shape[1]:,}")

    X_tr = hstack([X_tr_tf, csr_matrix(C_tr)])
    X_te = hstack([X_te_tf, csr_matrix(C_te)])
    print(f"  Combined features: {X_tr.shape[1]:,}  (tfidf + 3 context)")

    print(f"\nTraining LightGBM (n_estimators={args.n_estimators})...")
    model = LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=args.seed,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_tr, Y_tr)
    print(f"  Training: {time.time()-t0:.1f}s")

    print("\nEvaluating...")
    best_t, best_f1, probs = tune_threshold(model, X_te, Y_te)
    Y_pred = (probs >= best_t).astype(int)
    cr     = classification_report(Y_te, Y_pred, target_names=["non-FP", "FP"],
                                   output_dict=True, zero_division=0)
    print(f"  P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}  "
          f"F1={best_f1:.4f}  @t={best_t:.2f}")

    ctx_names   = ["script_density", "script_pred_fp_log", "script_size_log"]
    n_tfidf     = X_tr_tf.shape[1]
    ctx_imp     = {}
    print("\nContext feature importances:")
    for k, name in enumerate(ctx_names):
        imp = int(model.feature_importances_[n_tfidf + k])
        ctx_imp[name] = imp
        print(f"  {name:25s}: {imp}")

    metrics = {
        "f1":               best_f1,
        "precision":        cr["FP"]["precision"],
        "recall":           cr["FP"]["recall"],
        "threshold":        best_t,
        "train_size":       len(X_tr_raw),
        "test_size":        len(X_te_raw),
        "n_tfidf_features": n_tfidf,
        "context_feature_importances": ctx_imp,
        "baseline_f1":      0.853,
        "delta":            round(best_f1 - 0.853, 4),
        "binary":           True,
        "oracle":           False,
    }
    with open(os.path.join(args.out, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {args.out}/")
    print(f"F1: {best_f1:.4f}  (baseline 0.853, delta {best_f1 - 0.853:+.4f})")


if __name__ == "__main__":
    main()
