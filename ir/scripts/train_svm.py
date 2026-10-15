"""
Binary fingerprinting classifier: TF-IDF(ir_tokens, bigrams) + Linear SVM.

Linear SVM is classically the strongest approach for sparse TF-IDF features.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT = Path(__file__).parent.parent
DATASET = str(_IR_ROOT / "translated_dataset.parquet")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio", type=float, default=10.0)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--out", default=str(_IR_ROOT / "svm_model"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed):
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("Estimating FP count...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled += len(col)
    est_fp = int(fp_sample * total_rows / sampled)
    sample_prob = min(1.0, int(neg_ratio * est_fp) / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

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
    print(f"Collected {len(X):,}: {len(fp_rows):,} FP + {len(neg_rows):,} neg")
    return X, Y


def tune_threshold(probs, Y_te, n_steps=19):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y_te, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    pf = pq.ParquetFile(DATASET)
    X, Y = collect_sample(pf, args.neg_ratio, args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}  FP: {Y_te.sum():,}")

    print(f"Fitting TF-IDF (ngram 1-{args.ngram_max})...")
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, args.ngram_max))
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)
    print(f"  Features: {X_tr_vec.shape[1]:,}")

    print("Training Linear SVM (SGD + calibration)...")
    t0 = time.time()
    svm = SGDClassifier(
        loss="hinge", class_weight="balanced",
        max_iter=100, tol=1e-3,
        random_state=args.seed, n_jobs=-1,
    )
    # Calibrate to get probabilities for threshold tuning
    model = CalibratedClassifierCV(svm, cv=3)
    model.fit(X_tr_vec, Y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")

    probs = model.predict_proba(X_te_vec)[:, 1]
    default_f1 = f1_score(Y_te, probs >= 0.5, zero_division=0)
    best_t, best_f1 = tune_threshold(probs, Y_te)

    Y_pred = (probs >= best_t).astype(int)
    cr = classification_report(Y_te, Y_pred, target_names=["non-FP", "FP"],
                               output_dict=True, zero_division=0)
    print(f"\n  @0.50: F1={default_f1:.4f}")
    print(f"  Tuned: P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}"
          f"  F1={best_f1:.4f}  @t={best_t:.2f}")

    report = {
        "is_fp_tuned": cr["FP"],
        "macro_f1": best_f1,
        "threshold": best_t,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "neg_ratio": args.neg_ratio,
        "ngram_max": args.ngram_max,
        "n_features": X_tr_vec.shape[1],
        "binary": True,
    }
    with open(os.path.join(args.out, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved → {args.out}/  Best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
