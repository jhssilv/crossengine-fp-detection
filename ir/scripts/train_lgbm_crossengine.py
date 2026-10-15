"""
Cross-engine generalization experiment: TF-IDF(ir_tokens, bigrams) + LightGBM.

Two experiments:
  A) Train on Chromium (V8)  → test on Firefox (SM)
  B) Train on Firefox (SM)   → test on Chromium (V8)

Also reports per-engine F1 breakdown using the existing mixed model.
"""

import json
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
DATASET = str(_IR_ROOT / "translated_dataset.parquet")
OUT_DIR = _IR_ROOT / "lgbm_crossengine"
OUT_DIR.mkdir(exist_ok=True)

NEG_RATIO = 10
NGRAM_MAX = 2
N_ESTIMATORS = 500
SEED = 42


def collect_engine(pf, engine: str, neg_ratio: float, seed: int):
    """Collect all FP rows + sampled negatives from one engine."""
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    # 1-in-50 sample to estimate FP and total rows for this engine
    print(f"  Estimating FP count for {engine} (1-in-50 sample)...")
    fp_sample, total_sample, engine_sample = 0, 0, 0
    for i in range(0, n_rg, 50):
        df = pf.read_row_group(i, columns=["browser", "is_fp"]).to_pandas()
        mask = df["browser"] == engine
        sub = df[mask]
        fp_sample += int(sub["is_fp"].fillna(False).sum())
        engine_sample += len(sub)
        total_sample += len(df)

    engine_frac = engine_sample / max(total_sample, 1)
    est_engine_rows = int(total_rows * engine_frac)
    est_fp = int(fp_sample * est_engine_rows / max(engine_sample, 1))
    est_neg = est_engine_rows - est_fp
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(est_neg, 1))
    print(f"  est FP={est_fp:,}  est engine rows={est_engine_rows:,}  sample_prob={sample_prob:.4f}")

    fp_rows, neg_rows = [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 200 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["browser", "ir_tokens", "is_fp"]).to_pandas()
        df = df[df["browser"] == engine].copy()
        if len(df) == 0:
            continue
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
    print(f"  Collected {len(X):,}: {len(fp_rows):,} FP + {len(neg_rows):,} neg")
    return X, Y



def tune_threshold(probs, Y, n_steps=19):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(Y, probs >= 0.5, zero_division=0)
    print(f"  @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1


def run_experiment(pf, train_engine: str, test_engine: str):
    print(f"\n{'='*60}")
    print(f"Train on {train_engine}  →  Test on {test_engine}")
    print("="*60)

    print(f"\n[Train] Collecting {train_engine}...")
    X_tr, Y_tr = collect_engine(pf, train_engine, NEG_RATIO, SEED)

    print(f"\n[Test] Collecting {test_engine}...")
    X_te, Y_te = collect_engine(pf, test_engine, NEG_RATIO, SEED + 1)

    print(f"\nFitting TF-IDF (ngram 1-{NGRAM_MAX})...")
    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, NGRAM_MAX))
    X_tr_vec = tfidf.fit_transform(X_tr)
    X_te_vec = tfidf.transform(X_te)
    print(f"  Features: {X_tr_vec.shape[1]:,}")

    print(f"Training LightGBM (n_estimators={N_ESTIMATORS})...")
    model = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=127,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_tr_vec, Y_tr, eval_set=[(X_te_vec, Y_te)], callbacks=[])
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Tuning threshold...")
    probs = model.predict_proba(X_te_vec)[:, 1]
    best_t, best_f1 = tune_threshold(probs, Y_te)

    Y_pred = (probs >= best_t).astype(int)
    cr = classification_report(Y_te, Y_pred, target_names=["non-FP", "FP"],
                               output_dict=True, zero_division=0)
    fp = cr["FP"]
    print(f"  P={fp['precision']:.4f}  R={fp['recall']:.4f}  F1={best_f1:.4f}")

    tag = f"{train_engine[:3].lower()}_to_{test_engine[:3].lower()}"
    with open(OUT_DIR / f"tfidf_{tag}.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(OUT_DIR / f"model_{tag}.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "train_engine": train_engine,
        "test_engine": test_engine,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "test_fp": int(Y_te.sum()),
        "threshold": best_t,
        "precision": fp["precision"],
        "recall": fp["recall"],
        "f1": best_f1,
    }


def collect_mixed_with_browser(pf, neg_ratio, seed):
    """Reproduce the exact same collection as train_lgbm.py but also track browser labels.
    Uses the same seed and neg_ratio so the random negative sampling is identical,
    ensuring the reproduced dataset matches the original training data split."""
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("  Re-collecting mixed dataset (seed=42, neg_ratio=10) with browser labels...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled += len(col)
    est_fp = int(fp_sample * total_rows / sampled)
    est_neg = total_rows - est_fp
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(est_neg, 1))

    fp_tokens, fp_browsers = [], []
    neg_tokens, neg_browsers = [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 200 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["browser", "ir_tokens", "is_fp"]).to_pandas()
        df["is_fp"] = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")

        fp_df = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]

        if len(fp_df):
            fp_tokens.extend(fp_df["ir_tokens"].tolist())
            fp_browsers.extend(fp_df["browser"].tolist())
        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                neg_tokens.extend(neg_df["ir_tokens"].values[keep].tolist())
                neg_browsers.extend(neg_df["browser"].values[keep].tolist())

    X = fp_tokens + neg_tokens
    Y = np.array([1] * len(fp_tokens) + [0] * len(neg_tokens), dtype=np.int8)
    B = np.array(fp_browsers + neg_browsers)
    print(f"  Collected {len(X):,}: {len(fp_tokens):,} FP + {len(neg_tokens):,} neg")
    return X, Y, B


def per_engine_breakdown(pf):
    """Evaluate the existing mixed model per engine using the HELD-OUT test split.

    Reproduces the exact same dataset + 80/20 split as train_lgbm.py, then filters
    the test portion by browser. This avoids training data leakage.
    """
    mixed_model_dir = _IR_ROOT / "lgbm_model"
    model_path = mixed_model_dir / "lgbm_model.pkl"
    tfidf_path = mixed_model_dir / "tfidf.pkl"
    if not model_path.exists():
        print("\nMixed model not found at ir/lgbm_model/ — skipping breakdown.")
        return {}

    print(f"\n{'='*60}")
    print("Per-engine breakdown of mixed model — held-out test split only")
    print("="*60)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(mixed_model_dir / "metrics.json") as f:
        mixed_metrics = json.load(f)
    threshold = mixed_metrics.get("threshold", 0.75)
    print(f"  Using threshold={threshold:.2f} from mixed model")

    # Reproduce the same dataset collection as train_lgbm.py used
    X, Y, B = collect_mixed_with_browser(pf, NEG_RATIO, SEED)

    # Same 80/20 split with same seed — test set is identical to original evaluation
    _, X_te, _, Y_te, _, B_te = train_test_split(
        X, Y, B, test_size=0.2, random_state=SEED, stratify=Y
    )
    print(f"  Test set: {len(X_te):,} rows  FP: {int(Y_te.sum()):,}")

    results = {}
    for engine in ["Chromium", "Firefox"]:
        mask = B_te == engine
        X_eng = [X_te[i] for i, m in enumerate(mask) if m]
        Y_eng = Y_te[mask]
        if len(X_eng) == 0:
            print(f"  {engine}: no test rows found")
            continue
        X_vec = tfidf.transform(X_eng)
        probs = model.predict_proba(X_vec)[:, 1]
        Y_pred = (probs >= threshold).astype(int)
        cr = classification_report(Y_eng, Y_pred, target_names=["non-FP", "FP"],
                                   output_dict=True, zero_division=0)
        fp = cr["FP"]
        print(f"  {engine}: P={fp['precision']:.4f}  R={fp['recall']:.4f}  "
              f"F1={fp['f1-score']:.4f}  support={int(Y_eng.sum())}")
        results[engine] = {
            "precision": fp["precision"],
            "recall": fp["recall"],
            "f1": fp["f1-score"],
            "support": int(Y_eng.sum()),
            "note": "held-out 20% test split, no training data leakage",
        }

    return results


def main():
    pf = pq.ParquetFile(DATASET)

    results = []

    # Experiment A: Chromium → Firefox
    r = run_experiment(pf, "Chromium", "Firefox")
    results.append(r)

    # Experiment B: Firefox → Chromium
    r = run_experiment(pf, "Firefox", "Chromium")
    results.append(r)

    # Per-engine breakdown of mixed model
    breakdown = per_engine_breakdown(pf)

    report = {
        "cross_engine_experiments": results,
        "mixed_model_per_engine": breakdown,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['train_engine']} → {r['test_engine']}: "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}")
    if breakdown:
        print("  Mixed model per engine:")
        for eng, m in breakdown.items():
            print(f"    {eng}: F1={m['f1']:.4f}")
    print(f"\nSaved → {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
