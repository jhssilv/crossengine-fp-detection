"""
Sprint 7 — RF multilabel baseline.

Streams ir/translated_dataset.parquet, keeps ALL FP rows plus a random
subsample of non-FP rows (neg_ratio × n_fp), trains TF-IDF + RF, reports
per-class and macro/micro metrics, saves model and JSON report.

Usage:
    python ir/scripts/train_rf.py [--neg-ratio 10] [--n-estimators 200] [--out ir/rf_model]
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

LABEL_COLS = [
    "label_canvas",
    "label_audio",
    "label_webrtc",
    "label_font",
    "label_webgl",
    "label_screen",
    "label_navigator",
]

_IR_ROOT = Path(__file__).parent.parent
DATASET = str(_IR_ROOT / "translated_dataset.parquet")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio", type=float, default=10.0,
                   help="Non-FP rows kept per FP row (default 10)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--binary", action="store_true",
                   help="Train binary is_fp classifier instead of 7-label multilabel")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed, binary=False):
    """Single-pass stream: keep all FP rows + Bernoulli-sample non-FP rows."""
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    # --- estimate n_fp ---
    print("Estimating FP count from 1-in-50 row group sample...")
    fp_sample = 0
    sampled = 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].sum())
        sampled += len(col)
    est_fp = int(fp_sample * total_rows / sampled)
    est_neg = total_rows - est_fp
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(est_neg, 1))
    print(f"  Estimated FP rows: {est_fp:,}")
    print(f"  Target non-FP sample: {target_neg:,}  (sample_prob={sample_prob:.4f})")

    # --- full pass ---
    fp_rows, neg_rows = [], []
    fp_labels, neg_labels = [], []

    cols = ["ir_tokens", "is_fp"] + ([] if binary else LABEL_COLS)
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Row group {i}/{n_rg}  ({elapsed:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=cols).to_pandas()

        fp_mask = df["is_fp"].values.astype(bool)
        fp_df = df[fp_mask]
        neg_df = df[~fp_mask]

        if len(fp_df):
            fp_rows.extend(fp_df["ir_tokens"].tolist())
            if binary:
                fp_labels.append(np.ones(len(fp_df), dtype=np.int8))
            else:
                fp_labels.append(fp_df[LABEL_COLS].values.astype(np.int8))

        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                neg_rows.extend(neg_df["ir_tokens"].values[keep].tolist())
                if binary:
                    neg_labels.append(np.zeros(int(keep.sum()), dtype=np.int8))
                else:
                    neg_labels.append(neg_df[LABEL_COLS].values[keep].astype(np.int8))

    n_labels = 1 if binary else len(LABEL_COLS)
    empty_shape = (0,) if binary else (0, n_labels)
    Y_fp = np.concatenate(fp_labels) if fp_labels else np.empty(empty_shape, np.int8)
    Y_neg = np.concatenate(neg_labels) if neg_labels else np.empty(empty_shape, np.int8)

    X = fp_rows + neg_rows
    Y = np.concatenate([Y_fp, Y_neg], axis=0)
    print(f"Collected {len(X):,} rows: {len(fp_rows):,} FP + {len(neg_rows):,} non-FP")
    return X, Y


def main():
    args = parse_args()
    if args.out is None:
        args.out = str(_IR_ROOT / ("rf_model_binary" if args.binary else "rf_model"))
    os.makedirs(args.out, exist_ok=True)

    print(f"Mode: {'binary' if args.binary else 'multilabel'}")
    pf = pq.ParquetFile(DATASET)

    # --- collect sample ---
    X, Y = collect_sample(pf, args.neg_ratio, args.seed, binary=args.binary)

    # --- train/test split ---
    is_fp = Y if args.binary else Y.any(axis=1)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=is_fp
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")

    # --- TF-IDF ---
    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(sublinear_tf=True)
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)
    print(f"  Vocab size: {len(tfidf.vocabulary_)}")

    # --- RF ---
    print(f"Training RF (n_estimators={args.n_estimators})...")
    if args.binary:
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )
    else:
        rf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=args.seed,
            ),
            n_jobs=1,
        )
    t0 = time.time()
    rf.fit(X_tr_vec, Y_tr)
    print(f"  Training done in {time.time()-t0:.1f}s")

    # --- evaluate ---
    Y_pred = rf.predict(X_te_vec)
    report = {}

    if args.binary:
        cr = classification_report(
            Y_te, Y_pred, target_names=["non-FP", "FP"],
            output_dict=True, zero_division=0,
        )
        p = cr["FP"]["precision"]
        r = cr["FP"]["recall"]
        f = cr["FP"]["f1-score"]
        print(f"  {'is_fp':20s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  support={int(Y_te.sum())}")
        report["is_fp"] = cr["FP"]
        f1_macro = f1_micro = f
    else:
        for i, col in enumerate(LABEL_COLS):
            cr = classification_report(
                Y_te[:, i], Y_pred[:, i],
                target_names=["neg", col],
                output_dict=True,
                zero_division=0,
            )
            report[col] = cr[col]
            support = int(Y_te[:, i].sum())
            p = cr[col]["precision"]
            r = cr[col]["recall"]
            f = cr[col]["f1-score"]
            print(f"  {col:20s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  support={support}")
        f1_macro = f1_score(Y_te, Y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(Y_te, Y_pred, average="micro", zero_division=0)

    print(f"\n  macro-F1={f1_macro:.3f}  micro-F1={f1_micro:.3f}")
    report["macro_f1"] = f1_macro
    report["micro_f1"] = f1_micro
    report["train_size"] = len(X_tr)
    report["test_size"] = len(X_te)
    report["neg_ratio"] = args.neg_ratio
    report["n_estimators"] = args.n_estimators
    report["vocab_size"] = len(tfidf.vocabulary_)
    report["binary"] = args.binary

    # --- save ---
    model_path = os.path.join(args.out, "rf_model.pkl")
    tfidf_path = os.path.join(args.out, "tfidf.pkl")
    report_path = os.path.join(args.out, "metrics.json")

    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model → {model_path}")
    print(f"Saved TF-IDF → {tfidf_path}")
    print(f"Saved metrics → {report_path}")


if __name__ == "__main__":
    main()
