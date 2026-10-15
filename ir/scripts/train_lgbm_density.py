"""
Step 2 of the full-dataset script density pipeline.

Uses ir/script_density.json (computed by compute_script_density.py over ALL
80M functions) as an additional feature alongside TF-IDF bigrams.

Unlike the sparse-sampling attempt, script_fp_density here is computed from
the full function population of each script — not just the ~0.56% that lands
in the training sample — so the signal is reliable.

Usage:
    python ir/scripts/train_lgbm_density.py [--neg-ratio 10] [--out ir/lgbm_model_density]
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
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT     = Path(__file__).parent.parent
DATASET      = str(_IR_ROOT / "translated_dataset.parquet")
DENSITY_FILE = _IR_ROOT / "script_density.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio",    type=float, default=10.0)
    p.add_argument("--ngram-max",    type=int,   default=2)
    p.add_argument("--n-estimators", type=int,   default=500)
    p.add_argument("--out",          default=None)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed, density):
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
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    tokens_list, density_list, labels_list = [], [], []
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

        fp_df  = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]

        for sub, label in [(fp_df, 1), (neg_df, 0)]:
            if len(sub) == 0:
                continue
            if label == 0:
                keep = np.array([rng.random() < sample_prob for _ in range(len(sub))])
                if not keep.any():
                    continue
                sub = sub.iloc[keep]

            toks = sub["ir_tokens"].tolist()
            urls = sub["script_url"].tolist()
            dens = [density.get(u, 0.0) for u in urls]

            tokens_list.extend(toks)
            density_list.extend(dens)
            labels_list.extend([label] * len(toks))

    X = tokens_list
    D = np.array(density_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int8)
    print(f"Collected {len(X):,}: {Y.sum():,} FP + {(Y==0).sum():,} non-FP")
    return X, D, Y


def tune_threshold(model, X_vec, Y, n_steps=19):
    probs = model.predict_proba(X_vec)[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(Y, probs >= 0.5, zero_division=0)
    print(f"  Threshold: @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1


def main():
    args = parse_args()
    if args.out is None:
        args.out = str(_IR_ROOT / "lgbm_model_density")
    os.makedirs(args.out, exist_ok=True)

    print("Loading script density map...")
    with open(DENSITY_FILE) as f:
        density = json.load(f)
    print(f"  {len(density):,} scripts loaded")

    fp_scripts = sum(1 for v in density.values() if v >= 0.5)
    print(f"  Scripts with mean p_fp >= 0.5: {fp_scripts:,}")

    pf = pq.ParquetFile(DATASET)
    X, D, Y = collect_sample(pf, args.neg_ratio, args.seed, density)

    X_tr, X_te, D_tr, D_te, Y_tr, Y_te = train_test_split(
        X, D, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {Y_tr.sum():,}  Test FP: {Y_te.sum():,}")

    print(f"\nDensity signal check:")
    print(f"  Train FP   mean density: {D_tr[Y_tr==1].mean():.4f}")
    print(f"  Train non-FP mean density: {D_tr[Y_tr==0].mean():.4f}")
    print(f"  Test  FP   mean density: {D_te[Y_te==1].mean():.4f}")
    print(f"  Test  non-FP mean density: {D_te[Y_te==0].mean():.4f}")

    print(f"\nFitting TF-IDF (ngram 1-{args.ngram_max})...")
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, args.ngram_max))
    X_tr_tfidf = tfidf.fit_transform(X_tr)
    X_te_tfidf = tfidf.transform(X_te)
    print(f"  TF-IDF features: {X_tr_tfidf.shape[1]:,}")

    X_tr_aug = hstack([X_tr_tfidf, csr_matrix(D_tr.reshape(-1, 1))])
    X_te_aug = hstack([X_te_tfidf, csr_matrix(D_te.reshape(-1, 1))])
    print(f"  Combined features: {X_tr_aug.shape[1]:,}")

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
    model.fit(X_tr_aug, Y_tr, eval_set=[(X_te_aug, Y_te)], callbacks=[])
    print(f"  Training: {time.time()-t0:.1f}s")

    print("\nEvaluating...")
    best_t, best_f1 = tune_threshold(model, X_te_aug, Y_te)
    probs = model.predict_proba(X_te_aug)[:, 1]
    Y_pred = (probs >= best_t).astype(int)
    cr = classification_report(Y_te, Y_pred, target_names=["non-FP", "FP"],
                               output_dict=True, zero_division=0)
    print(f"  P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}  F1={best_f1:.4f}")

    # Density feature importance
    n_tfidf = X_tr_tfidf.shape[1]
    density_imp = model.feature_importances_[n_tfidf]
    print(f"\n  script_fp_density split gain: {density_imp:.1f}")

    report = {
        "f1":              best_f1,
        "precision":       cr["FP"]["precision"],
        "recall":          cr["FP"]["recall"],
        "threshold":       best_t,
        "density_importance": float(density_imp),
        "train_size":      len(X_tr),
        "test_size":       len(X_te),
        "n_tfidf_features": n_tfidf,
        "binary":          True,
    }
    with open(os.path.join(args.out, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved → {args.out}/")
    print(f"Best F1: {best_f1:.4f} at threshold={best_t:.2f}")
    print(f"Baseline was 0.857 — delta: {best_f1-0.857:+.4f}")


if __name__ == "__main__":
    main()
