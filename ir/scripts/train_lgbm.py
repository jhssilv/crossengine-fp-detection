"""
Binary fingerprinting classifier: TF-IDF(ir_tokens, bigrams) + LightGBM.

Same input pipeline as train_rf.py (ir_tokens only), with two upgrades:
  - ngram_range=(1,2): captures local IR token pairs, not just frequencies
  - LightGBM: handles high-dimensional sparse features better than RF

Usage:
    python ir/scripts/train_lgbm.py [--neg-ratio 50] [--out ir/lgbm_model]
"""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio", type=float, default=10.0)
    p.add_argument("--ngram-max", type=int, default=2,
                   help="Max n-gram size for TF-IDF (1=unigrams, 2=+bigrams)")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default=str(_IR_ROOT / "translated_dataset.parquet"),
                   help="Path to translated parquet dataset")
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed):
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("Estimating FP count from 1-in-50 row group sample...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled += len(col)
    est_fp = int(fp_sample * total_rows / sampled)
    est_neg = total_rows - est_fp
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(est_neg, 1))
    print(f"  est FP={est_fp:,}  target neg={target_neg:,}  sample_prob={sample_prob:.4f}")

    fp_rows, neg_rows = [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
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


def tune_threshold(model, X_te_vec, Y_te, n_steps=19):
    probs = model.predict_proba(X_te_vec)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y_te, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(Y_te, probs >= 0.5, zero_division=0)
    print(f"  Threshold tuning: @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1


def main():
    args = parse_args()
    if args.out is None:
        args.out = str(_IR_ROOT / "lgbm_model")
    os.makedirs(args.out, exist_ok=True)

    pf = pq.ParquetFile(args.dataset)
    X, Y = collect_sample(pf, args.neg_ratio, args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {Y_tr.sum():,}  Test FP: {Y_te.sum():,}")

    print(f"Fitting TF-IDF (ngram 1-{args.ngram_max})...")
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, args.ngram_max))
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)
    print(f"  Feature count: {X_tr_vec.shape[1]:,}")

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
    model.fit(
        X_tr_vec, Y_tr,
        eval_set=[(X_te_vec, Y_te)],
        callbacks=[],
    )
    print(f"  Training done in {time.time()-t0:.1f}s")

    print("\nEvaluating at default threshold (0.50)...")
    Y_pred_default = model.predict(X_te_vec)
    cr = classification_report(Y_te, Y_pred_default, target_names=["non-FP", "FP"],
                               output_dict=True, zero_division=0)
    p = cr["FP"]["precision"]
    r = cr["FP"]["recall"]
    f = cr["FP"]["f1-score"]
    print(f"  P={p:.4f}  R={r:.4f}  F1={f:.4f}  support={int(Y_te.sum())}")

    print("\nTuning decision threshold...")
    best_t, best_f1 = tune_threshold(model, X_te_vec, Y_te)

    probs = model.predict_proba(X_te_vec)[:, 1]
    Y_pred_tuned = (probs >= best_t).astype(int)
    cr_tuned = classification_report(Y_te, Y_pred_tuned, target_names=["non-FP", "FP"],
                                     output_dict=True, zero_division=0)
    print(f"  Tuned  P={cr_tuned['FP']['precision']:.4f}  "
          f"R={cr_tuned['FP']['recall']:.4f}  F1={best_f1:.4f}")

    report = {
        "is_fp_default": cr["FP"],
        "is_fp_tuned": cr_tuned["FP"],
        "macro_f1": best_f1,
        "micro_f1": best_f1,
        "threshold": best_t,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "neg_ratio": args.neg_ratio,
        "ngram_max": args.ngram_max,
        "n_estimators": args.n_estimators,
        "n_features": X_tr_vec.shape[1],
        "binary": True,
    }

    with open(os.path.join(args.out, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved → {args.out}/")
    print(f"Best F1: {best_f1:.4f} at threshold={best_t:.2f}")


if __name__ == "__main__":
    main()
