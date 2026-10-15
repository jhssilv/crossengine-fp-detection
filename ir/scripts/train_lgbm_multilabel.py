"""
Multilabel fingerprinting classifier: TF-IDF(ir_tokens, bigrams) + LightGBM.

One LGBMClassifier per label (via MultiOutputClassifier) — same pipeline as the
binary model, extended to predict all 7 fingerprinting categories simultaneously.
"""

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
from sklearn.multioutput import MultiOutputClassifier

_IR_ROOT = Path(__file__).parent.parent
DATASET = str(_IR_ROOT / "translated_dataset.parquet")

LABEL_COLS = [
    "label_canvas", "label_audio", "label_webrtc",
    "label_font", "label_webgl", "label_screen", "label_navigator",
]

NGRAM_MAX = 2
NEG_RATIO = 10
N_ESTIMATORS = 300
SEED = 42
OUT = str(_IR_ROOT / "lgbm_model_multilabel")


def collect_sample(pf, neg_ratio, seed):
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
    print(f"  est FP={est_fp:,}  target neg={target_neg:,}  sample_prob={sample_prob:.4f}")

    fp_tokens, fp_labels = [], []
    neg_tokens = []
    t0 = time.time()
    read_cols = ["ir_tokens", "is_fp"] + LABEL_COLS

    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=read_cols).to_pandas()
        df["is_fp"] = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")
        for col in LABEL_COLS:
            df[col] = df[col].fillna(False).astype(bool)

        fp_df = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]

        if len(fp_df):
            fp_tokens.extend(fp_df["ir_tokens"].tolist())
            fp_labels.extend(fp_df[LABEL_COLS].values.tolist())

        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                neg_tokens.extend(neg_df["ir_tokens"].values[keep].tolist())

    neg_labels = [[False] * len(LABEL_COLS)] * len(neg_tokens)

    X = fp_tokens + neg_tokens
    Y = np.array(fp_labels + neg_labels, dtype=np.int8)
    print(f"Collected {len(X):,}: {len(fp_tokens):,} FP + {len(neg_tokens):,} neg")
    return X, Y


def tune_thresholds(models, tfidf, X_te, Y_te, n_steps=19):
    """Per-label threshold tuning. Returns dict label→threshold."""
    X_vec = tfidf.transform(X_te)
    thresholds = {}
    for idx, (label, clf) in enumerate(zip(LABEL_COLS, models.estimators_)):
        probs = clf.predict_proba(X_vec)[:, 1]
        Y_col = Y_te[:, idx]
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, n_steps):
            f = f1_score(Y_col, probs >= t, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds[label] = best_t
        default_f1 = f1_score(Y_col, probs >= 0.5, zero_division=0)
        support = int(Y_col.sum())
        print(f"  {label:20s}  @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}  support={support}")
    return thresholds


def evaluate(models, tfidf, thresholds, X_te, Y_te):
    X_vec = tfidf.transform(X_te)
    per_label = {}
    for idx, (label, clf) in enumerate(zip(LABEL_COLS, models.estimators_)):
        probs = clf.predict_proba(X_vec)[:, 1]
        Y_col = Y_te[:, idx]
        t = thresholds[label]
        Y_pred = (probs >= t).astype(int)
        cr = classification_report(Y_col, Y_pred, target_names=["non-FP", "FP"],
                                   output_dict=True, zero_division=0)
        per_label[label] = cr["FP"]
    macro_f1 = float(np.mean([v["f1-score"] for v in per_label.values()]))
    return per_label, macro_f1


def main():
    os.makedirs(OUT, exist_ok=True)
    pf = pq.ParquetFile(DATASET)
    X, Y = collect_sample(pf, NEG_RATIO, SEED)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=SEED, stratify=Y[:, 0]
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {Y_tr[:, 0].sum():,}  Test FP: {Y_te[:, 0].sum():,}")

    print(f"\nFitting TF-IDF (ngram 1-{NGRAM_MAX})...")
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, NGRAM_MAX))
    X_tr_vec = tfidf.fit_transform(X_tr)
    print(f"  Features: {X_tr_vec.shape[1]:,}")

    print(f"\nTraining MultiOutputClassifier(LightGBM, n_estimators={N_ESTIMATORS})...")
    base = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=63,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    model = MultiOutputClassifier(base, n_jobs=-1)
    t0 = time.time()
    model.fit(X_tr_vec, Y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nTuning per-label thresholds...")
    X_te_vec = tfidf.transform(X_te)
    thresholds = tune_thresholds(model, tfidf, X_te, Y_te)

    print("\nFinal evaluation...")
    per_label, macro_f1 = evaluate(model, tfidf, thresholds, X_te, Y_te)
    print(f"\n  Macro F1: {macro_f1:.4f}")
    for label, m in per_label.items():
        print(f"  {label:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1-score']:.4f}")

    report = {
        "per_label": per_label,
        "thresholds": thresholds,
        "macro_f1": macro_f1,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "neg_ratio": NEG_RATIO,
        "ngram_max": NGRAM_MAX,
        "n_estimators": N_ESTIMATORS,
        "n_features": X_tr_vec.shape[1],
    }
    with open(os.path.join(OUT, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(OUT, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
