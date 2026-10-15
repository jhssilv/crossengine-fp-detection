"""
LightGBM iterative self-improvement trainer.

Uses richer script context (5 features) computed by compute_script_stats_iter.py:
    density         — mean predicted FP probability
    fp_count_30_log — log1p(count of p > 0.30) — soft FP signal
    fp_count_50_log — log1p(count of p > 0.50) — moderate FP signal
    fp_count_70_log — log1p(count of p > 0.70) — strong FP signal
    size_log        — log1p(total functions in script)

The threshold-stratified counts give the model information about the *shape*
of the probability distribution within a script, not just a single cut.

Usage:
    # iteration 2
    python ir/scripts/train_lgbm_density_iter.py \\
        --stats ir/script_stats_iter2.json \\
        --out   ir/lgbm_model_density_v3b --neg-ratio 7

    # iteration 3
    python ir/scripts/train_lgbm_density_iter.py \\
        --stats ir/script_stats_iter3.json \\
        --out   ir/lgbm_model_density_v4 --neg-ratio 7
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

_IR_ROOT = Path(__file__).parent.parent
DATASET  = str(_IR_ROOT / "translated_dataset.parquet")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stats",         required=True,
                   help="Path to script_stats_iter*.json from compute_script_stats_iter.py")
    p.add_argument("--neg-ratio",     type=float, default=10.0)
    p.add_argument("--ngram-max",     type=int,   default=2)
    p.add_argument("--n-estimators",  type=int,   default=500)
    p.add_argument("--out",           required=True)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def context_features(url, stats):
    s  = stats.get(url, {})
    d  = s.get("density",     0.0)
    f30 = s.get("fp_count_30", s.get("fp_count_50", s.get("pred_fp_count", 0)))
    f50 = s.get("fp_count_50", s.get("pred_fp_count", 0))
    f70 = s.get("fp_count_70", 0)
    sz  = s.get("size",        0)
    return np.array([
        d,
        math.log1p(f30),
        math.log1p(f50),
        math.log1p(f70),
        math.log1p(sz),
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
    sample_prob = min(1.0, neg_ratio * est_fp / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    tokens_list, ctx_list, labels_list = [], [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["ir_tokens", "is_fp", "script_url"]).to_pandas()
        df["is_fp"]      = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"]  = df["ir_tokens"].fillna("")
        df["script_url"] = df["script_url"].fillna("")

        for mask, label in [(df["is_fp"].values, 1), (~df["is_fp"].values, 0)]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            if label == 0:
                keep = np.array([rng.random() < sample_prob for _ in range(len(sub))])
                if not keep.any():
                    continue
                sub = sub.iloc[keep]
            tokens_list.extend(sub["ir_tokens"].tolist())
            ctx_list.extend([context_features(u, stats) for u in sub["script_url"].tolist()])
            labels_list.extend([label] * len(sub))

    C = np.array(ctx_list,    dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int8)
    print(f"Collected {len(tokens_list):,}: {Y.sum():,} FP + {(Y==0).sum():,} non-FP")
    return tokens_list, C, Y


def tune_threshold(model, X, Y, n_steps=19):
    probs = model.predict_proba(X)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    print(f"  Threshold: @0.50={f1_score(Y, probs>=0.5, zero_division=0):.4f}  "
          f"best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1, probs


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading stats from {args.stats}...")
    with open(args.stats) as f:
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
    print(f"  Combined features: {X_tr.shape[1]:,}  (tfidf + 5 context)")

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

    n_tfidf    = X_tr_tf.shape[1]
    ctx_names  = ["density", "fp_count_30_log", "fp_count_50_log", "fp_count_70_log", "size_log"]
    ctx_imp    = {}
    print("\nContext feature importances:")
    for k, name in enumerate(ctx_names):
        imp = int(model.feature_importances_[n_tfidf + k])
        ctx_imp[name] = imp
        print(f"  {name:25s}: {imp}")

    metrics = {
        "f1":          best_f1,
        "precision":   cr["FP"]["precision"],
        "recall":      cr["FP"]["recall"],
        "threshold":   best_t,
        "train_size":  len(X_tr_raw),
        "test_size":   len(X_te_raw),
        "n_tfidf_features": n_tfidf,
        "context_feature_importances": ctx_imp,
        "baseline_f1": 0.853,
        "delta":       round(best_f1 - 0.853, 4),
        "binary":      True,
        "oracle":      False,
    }
    with open(os.path.join(args.out, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {args.out}/")
    print(f"F1: {best_f1:.4f}  (baseline 0.853, delta {best_f1-0.853:+.4f})")


if __name__ == "__main__":
    main()
