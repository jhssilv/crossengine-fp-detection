"""
Sprint 11, Tasks 1 + 2: structural features + script-level call-graph context.

Features added on top of the density model baseline (F1=0.853):

  Task 1 — structural (computed from ir_tokens, obfuscation-invariant):
    token_count_norm    log1p(token count) / log1p(10000)
    call_density        CALL_* tokens / total
    branch_ratio        JMP_* + SWITCH_* tokens / total
    prop_access_ratio   GET/SET _NAMED/KEYED_PROP tokens / total
    unique_token_ratio  distinct token types used / 145
    seq_entropy_norm    Shannon entropy / log2(145)

  Task 2 — script-level context (from compute_script_stats.py, ground-truth counts):
    script_fp_count_log   log1p(FP functions in same script)
    script_size_log       log1p(total functions in same script)
    script_gt_density     fp_count / total  (ground-truth ratio, vs model-predicted)

  Retained from density model:
    script_fp_density     mean predicted FP probability (from script_density.json)

Baseline: lgbm_model_density  F1=0.853

Usage:
    python ir/scripts/train_lgbm_sprint11.py [--neg-ratio 10] [--out ir/lgbm_model_sprint11]
"""

import argparse
import json
import math
import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT      = Path(__file__).parent.parent
DATASET       = str(_IR_ROOT / "translated_dataset.parquet")
DENSITY_FILE  = _IR_ROOT / "script_density.json"
STATS_FILE    = _IR_ROOT / "script_stats.json"
VOCAB_SIZE    = 145   # total IR tokens

CALL_TOKENS = frozenset([
    "CALL_FUNC", "CALL_FUNC_0", "CALL_FUNC_1", "CALL_FUNC_2", "CALL_FUNC_N",
    "CALL_ITER", "CALL_METHOD_0", "CALL_METHOD_1", "CALL_METHOD_2", "CALL_METHOD_N",
    "CALL_NEW", "CALL_RUNTIME", "CALL_SUPER",
])
BRANCH_TOKENS = frozenset([
    "JMP", "JMP_IF_FALSE", "JMP_IF_NOT_NULL", "JMP_IF_NOT_UNDEFINED",
    "JMP_IF_NULL", "JMP_IF_OBJ", "JMP_IF_TRUE", "JMP_IF_UNDEFINED",
    "JMP_LOOP", "JUMP_TARGET", "SWITCH", "SWITCH_CASE", "SWITCH_DEFAULT",
])
PROP_TOKENS = frozenset([
    "GET_NAMED_PROP", "GET_KEYED_PROP", "SET_NAMED_PROP", "SET_KEYED_PROP",
])
LOG_VOCAB    = math.log2(VOCAB_SIZE)
LOG_MAX_TOKS = math.log1p(10_000)


def structural_features(tok_str: str) -> np.ndarray:
    """Return 6 obfuscation-invariant features from a space-separated IR token string."""
    if not tok_str:
        return np.zeros(6, dtype=np.float32)
    toks = tok_str.split()
    n = len(toks)
    if n == 0:
        return np.zeros(6, dtype=np.float32)
    c = Counter(toks)
    call_cnt   = sum(c[t] for t in CALL_TOKENS   if t in c)
    branch_cnt = sum(c[t] for t in BRANCH_TOKENS if t in c)
    prop_cnt   = sum(c[t] for t in PROP_TOKENS   if t in c)
    entropy    = -sum((v / n) * math.log2(v / n) for v in c.values() if v > 0)
    return np.array([
        math.log1p(n) / LOG_MAX_TOKS,        # token_count_norm
        call_cnt   / n,                       # call_density
        branch_cnt / n,                       # branch_ratio
        prop_cnt   / n,                       # prop_access_ratio
        len(c)     / VOCAB_SIZE,              # unique_token_ratio
        entropy    / LOG_VOCAB,               # seq_entropy_norm
    ], dtype=np.float32)


def script_context_features(url: str, density: dict, stats: dict) -> np.ndarray:
    """Return 4 script-level context features for a given script_url."""
    pred_density = density.get(url, 0.0)
    s = stats.get(url, {})
    fp_cnt = s.get("fp_count", 0)
    total  = s.get("total", 0)
    gt_density = fp_cnt / total if total > 0 else 0.0
    return np.array([
        pred_density,                   # script_fp_density   (model-predicted)
        math.log1p(fp_cnt),             # script_fp_count_log (ground-truth)
        math.log1p(total),              # script_size_log
        gt_density,                     # script_gt_density
    ], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio",    type=float, default=10.0)
    p.add_argument("--ngram-max",    type=int,   default=2)
    p.add_argument("--n-estimators", type=int,   default=500)
    p.add_argument("--out",          default=None)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def collect_sample(pf, neg_ratio, seed, density, stats):
    rng = random.Random(seed)
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

    tokens_list, struct_list, ctx_list, labels_list = [], [], [], []
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
            struct_list.extend([structural_features(t) for t in toks])
            ctx_list.extend([script_context_features(u, density, stats) for u in urls])
            labels_list.extend([label] * len(toks))

    X     = tokens_list
    S     = np.array(struct_list, dtype=np.float32)   # [n, 6]
    C     = np.array(ctx_list,    dtype=np.float32)   # [n, 4]
    Y     = np.array(labels_list, dtype=np.int8)
    print(f"Collected {len(X):,}: {Y.sum():,} FP + {(Y==0).sum():,} non-FP")
    return X, S, C, Y


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
        args.out = str(_IR_ROOT / "lgbm_model_sprint11")
    os.makedirs(args.out, exist_ok=True)

    print("Loading script density map...")
    with open(DENSITY_FILE) as f:
        density = json.load(f)
    print(f"  {len(density):,} scripts")

    print("Loading script stats (ground-truth counts)...")
    if not STATS_FILE.exists():
        print(f"  ERROR: {STATS_FILE} not found.")
        print(f"  Run: python ir/scripts/compute_script_stats.py")
        return
    with open(STATS_FILE) as f:
        stats = json.load(f)
    print(f"  {len(stats):,} scripts")

    pf = pq.ParquetFile(DATASET)
    X, S, C, Y = collect_sample(pf, args.neg_ratio, args.seed, density, stats)

    X_tr, X_te, S_tr, S_te, C_tr, C_te, Y_tr, Y_te = train_test_split(
        X, S, C, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {Y_tr.sum():,}  Test FP: {Y_te.sum():,}")

    print(f"\nFitting TF-IDF (ngram 1-{args.ngram_max})...")
    tfidf    = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, args.ngram_max))
    X_tr_tf  = tfidf.fit_transform(X_tr)
    X_te_tf  = tfidf.transform(X_te)
    print(f"  TF-IDF features: {X_tr_tf.shape[1]:,}")

    X_tr_aug = hstack([X_tr_tf, csr_matrix(S_tr), csr_matrix(C_tr)])
    X_te_aug = hstack([X_te_tf, csr_matrix(S_te), csr_matrix(C_te)])
    print(f"  Combined features: {X_tr_aug.shape[1]:,}  "
          f"(tfidf={X_tr_tf.shape[1]}, structural=6, context=4)")

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
    probs  = model.predict_proba(X_te_aug)[:, 1]
    Y_pred = (probs >= best_t).astype(int)
    cr     = classification_report(Y_te, Y_pred, target_names=["non-FP", "FP"],
                                   output_dict=True, zero_division=0)
    print(f"  P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}  F1={best_f1:.4f}")

    # Feature importances for the 10 dense features
    n_tfidf = X_tr_tf.shape[1]
    struct_names = [
        "token_count_norm", "call_density", "branch_ratio",
        "prop_access_ratio", "unique_token_ratio", "seq_entropy_norm",
    ]
    ctx_names = [
        "script_fp_density", "script_fp_count_log", "script_size_log", "script_gt_density",
    ]
    dense_importances = {}
    for k, name in enumerate(struct_names + ctx_names):
        imp = model.feature_importances_[n_tfidf + k]
        dense_importances[name] = int(imp)
        print(f"    {name:30s}: {imp:.0f}")

    report = {
        "f1":        best_f1,
        "precision": cr["FP"]["precision"],
        "recall":    cr["FP"]["recall"],
        "threshold": best_t,
        "train_size": len(X_tr),
        "test_size":  len(X_te),
        "n_tfidf_features": n_tfidf,
        "dense_feature_importances": dense_importances,
        "baseline_density_f1": 0.853,
        "delta": round(best_f1 - 0.853, 4),
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
    print(f"Baseline was 0.853 — delta: {best_f1 - 0.853:+.4f}")


if __name__ == "__main__":
    main()
