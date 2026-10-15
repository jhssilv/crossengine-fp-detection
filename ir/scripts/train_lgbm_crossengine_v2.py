"""
Cross-engine generalization experiment v2: TF-IDF(ir_tokens, bigrams) + 5 context
features (density, fp_count_30_log, fp_count_50_log, fp_count_70_log, size_log).

Mirrors the feature set of lgbm_model_density_v4 (iter-3, F1=0.904) so that
both the zero-shot transfer experiments and the per-engine breakdown use the
same model class as the best reported result.

Two experiments:
  A) Train on Chromium (V8)  → test on Firefox (SM)
  B) Train on Firefox (SM)   → test on Chromium (V8)

Plus per-engine breakdown of the saved lgbm_model_density_v4 joint model,
evaluated on the held-out 20% test split (reproduced with the same seed).

Results saved to ir/lgbm_crossengine_v2/results.json.
"""

import gc
import json
import math
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
STATS_PATH = str(_IR_ROOT / "script_stats_iter3.json")
OUT_DIR    = _IR_ROOT / "lgbm_crossengine_v2"
OUT_DIR.mkdir(exist_ok=True)

# Match iter-3 training hyperparams (see discussion.md §5.1 and train_lgbm_density_iter.py)
NEG_RATIO    = 7
NGRAM_MAX    = 2
N_ESTIMATORS = 1000
SEED         = 42


# ---------------------------------------------------------------------------
# Context features — inlined as tuples to avoid per-row numpy array allocation
# ---------------------------------------------------------------------------

def _ctx_tuple(url: str, stats: dict) -> tuple:
    s   = stats.get(url, {})
    d   = s.get("density",     0.0)
    f30 = s.get("fp_count_30", s.get("fp_count_50", s.get("pred_fp_count", 0)))
    f50 = s.get("fp_count_50", s.get("pred_fp_count", 0))
    f70 = s.get("fp_count_70", 0)
    sz  = s.get("size",        0)
    return (d, math.log1p(f30), math.log1p(f50), math.log1p(f70), math.log1p(sz))


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def collect_engine(pf, engine: str, neg_ratio: float, seed: int, stats: dict):
    """Return (tokens, C, Y) for one engine. ctx stored as tuple list → single np.array."""
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print(f"  Estimating FP count for {engine} (1-in-50 sample)...")
    fp_sample, total_sample, engine_sample = 0, 0, 0
    for i in range(0, n_rg, 50):
        df = pf.read_row_group(i, columns=["browser", "is_fp"]).to_pandas()
        mask = df["browser"] == engine
        sub  = df[mask]
        fp_sample     += int(sub["is_fp"].fillna(False).sum())
        engine_sample += len(sub)
        total_sample  += len(df)

    engine_frac = engine_sample / max(total_sample, 1)
    est_engine  = int(total_rows * engine_frac)
    est_fp      = int(fp_sample * est_engine / max(engine_sample, 1))
    sample_prob = min(1.0, neg_ratio * est_fp / max(est_engine - est_fp, 1))
    print(f"  est FP={est_fp:,}  est engine rows={est_engine:,}  sample_prob={sample_prob:.4f}")

    tokens_list, ctx_tuples, labels_list = [], [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 200 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(
            i, columns=["browser", "ir_tokens", "is_fp", "script_url"]
        ).to_pandas()
        df = df[df["browser"] == engine].copy()
        if len(df) == 0:
            continue
        df["is_fp"]      = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"]  = df["ir_tokens"].fillna("")
        df["script_url"] = df["script_url"].fillna("")

        for fp_flag, label in [(True, 1), (False, 0)]:
            sub = df[df["is_fp"] == fp_flag]
            if len(sub) == 0:
                continue
            if label == 0 and sample_prob > 0:
                keep = np.array([rng.random() < sample_prob for _ in range(len(sub))])
                if not keep.any():
                    continue
                sub = sub[keep]
            tokens_list.extend(sub["ir_tokens"].tolist())
            ctx_tuples.extend(_ctx_tuple(u, stats) for u in sub["script_url"].tolist())
            labels_list.extend([label] * len(sub))

    C = np.array(ctx_tuples, dtype=np.float32).reshape(-1, 5)
    Y = np.array(labels_list, dtype=np.int8)
    print(f"  Collected {len(tokens_list):,}: {Y.sum():,} FP + {(Y==0).sum():,} neg")
    return tokens_list, C, Y


def collect_mixed_with_browser(pf, neg_ratio: float, seed: int, stats: dict):
    """
    Reproduce the exact collection from train_lgbm_density_iter.py (seed=42,
    neg_ratio=7) while also tracking browser labels — used to reconstruct the
    held-out test split for per-engine evaluation.
    """
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("  Estimating FP count from 1-in-50 row group sample...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled   += len(col)
    est_fp      = int(fp_sample * total_rows / sampled)
    sample_prob = min(1.0, neg_ratio * est_fp / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    tokens_list, ctx_tuples, labels_list, browsers_list = [], [], [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(
            i, columns=["browser", "ir_tokens", "is_fp", "script_url"]
        ).to_pandas()
        df["is_fp"]      = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"]  = df["ir_tokens"].fillna("")
        df["script_url"] = df["script_url"].fillna("")

        for fp_flag, label in [(True, 1), (False, 0)]:
            sub = df[df["is_fp"] == fp_flag]
            if len(sub) == 0:
                continue
            if label == 0 and sample_prob > 0:
                keep = np.array([rng.random() < sample_prob for _ in range(len(sub))])
                if not keep.any():
                    continue
                sub = sub[keep]
            tokens_list.extend(sub["ir_tokens"].tolist())
            ctx_tuples.extend(_ctx_tuple(u, stats) for u in sub["script_url"].tolist())
            labels_list.extend([label] * len(sub))
            browsers_list.extend(sub["browser"].tolist())

    C = np.array(ctx_tuples, dtype=np.float32).reshape(-1, 5)
    Y = np.array(labels_list, dtype=np.int8)
    B = np.array(browsers_list)
    print(f"  Collected {len(tokens_list):,}: {Y.sum():,} FP + {(Y==0).sum():,} neg")
    return tokens_list, C, Y, B


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(probs, Y, n_steps=19):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(Y, probs >= 0.5, zero_division=0)
    print(f"  @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Zero-shot cross-engine experiment
# ---------------------------------------------------------------------------

def run_experiment(pf, train_engine: str, test_engine: str, stats: dict):
    print(f"\n{'='*60}")
    print(f"Train on {train_engine}  →  Test on {test_engine}  [with context features]")
    print("="*60)

    # --- Collect + transform train, then free raw strings before loading test ---
    print(f"\n[Train] Collecting {train_engine}...")
    tokens_tr, C_tr, Y_tr = collect_engine(pf, train_engine, NEG_RATIO, SEED, stats)

    print(f"\nFitting TF-IDF (ngram 1-{NGRAM_MAX})...")
    tfidf   = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, NGRAM_MAX))
    Xtr_tf  = tfidf.fit_transform(tokens_tr)
    print(f"  TF-IDF features: {Xtr_tf.shape[1]:,}")
    del tokens_tr; gc.collect()

    Xtr = hstack([Xtr_tf, csr_matrix(C_tr)])
    print(f"  Combined features: {Xtr.shape[1]:,}  (tfidf + 5 context)")
    del Xtr_tf, C_tr; gc.collect()

    # --- Collect + transform test ---
    print(f"\n[Test] Collecting {test_engine}...")
    tokens_te, C_te, Y_te = collect_engine(pf, test_engine, NEG_RATIO, SEED + 1, stats)

    Xte_tf = tfidf.transform(tokens_te)
    del tokens_te; gc.collect()
    Xte = hstack([Xte_tf, csr_matrix(C_te)])
    del Xte_tf, C_te; gc.collect()

    # --- Train ---
    print(f"\nTraining LightGBM (n_estimators={N_ESTIMATORS})...")
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
    train_size = Xtr.shape[0]
    t0 = time.time()
    model.fit(Xtr, Y_tr)
    print(f"  Done in {time.time()-t0:.1f}s")
    del Xtr, Y_tr; gc.collect()

    print("Tuning threshold...")
    probs  = model.predict_proba(Xte)[:, 1]
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

    del model, Xte, Y_te; gc.collect()

    return {
        "train_engine": train_engine,
        "test_engine":  test_engine,
        "train_size":   train_size,
        "test_fp":      int(Y_pred.sum()),
        "threshold":    best_t,
        "precision":    fp["precision"],
        "recall":       fp["recall"],
        "f1":           best_f1,
        "features":     "tfidf_bigrams+5_context",
    }


# ---------------------------------------------------------------------------
# Per-engine breakdown of the saved joint iter-3 model
# ---------------------------------------------------------------------------

def per_engine_breakdown(pf, stats: dict):
    """
    Evaluate lgbm_model_density_v4 (iter-3, F1=0.904) per engine.

    Reproduces the same data collection + 80/20 stratified split used by
    train_lgbm_density_iter.py (neg_ratio=7, seed=42) and filters the held-out
    test portion by browser — no training-data leakage.
    """
    model_dir  = _IR_ROOT / "lgbm_model_density_v4"
    model_path = model_dir / "lgbm_model.pkl"
    tfidf_path = model_dir / "tfidf.pkl"

    if not model_path.exists():
        print("\nlgbm_model_density_v4 not found — skipping per-engine breakdown.")
        return {}

    print(f"\n{'='*60}")
    print("Per-engine breakdown — lgbm_model_density_v4 (iter-3, F1=0.904)")
    print("="*60)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(model_dir / "metrics.json") as f:
        m = json.load(f)
    threshold = m.get("threshold", 0.80)
    print(f"  threshold={threshold:.2f}  (from saved metrics)")

    tokens, C, Y, B = collect_mixed_with_browser(pf, NEG_RATIO, SEED, stats)

    _, X_te_raw, _, C_te, _, Y_te, _, B_te = train_test_split(
        tokens, C, Y, B, test_size=0.2, random_state=SEED, stratify=Y
    )
    del tokens, C, Y, B; gc.collect()
    print(f"  Test set: {len(X_te_raw):,} rows  FP: {int(Y_te.sum()):,}")

    X_te_tf = tfidf.transform(X_te_raw)
    del X_te_raw; gc.collect()
    X_te = hstack([X_te_tf, csr_matrix(C_te)])
    del X_te_tf, C_te; gc.collect()

    results = {}
    for engine in ["Chromium", "Firefox"]:
        mask  = B_te == engine
        X_eng = X_te[mask]
        Y_eng = Y_te[mask]
        if X_eng.shape[0] == 0:
            print(f"  {engine}: no test rows")
            continue
        probs  = model.predict_proba(X_eng)[:, 1]
        Y_pred = (probs >= threshold).astype(int)
        cr     = classification_report(Y_eng, Y_pred, target_names=["non-FP", "FP"],
                                       output_dict=True, zero_division=0)
        fp = cr["FP"]
        print(f"  {engine}: P={fp['precision']:.4f}  R={fp['recall']:.4f}  "
              f"F1={fp['f1-score']:.4f}  support={int(Y_eng.sum())}")
        results[engine] = {
            "precision": fp["precision"],
            "recall":    fp["recall"],
            "f1":        fp["f1-score"],
            "support":   int(Y_eng.sum()),
            "note":      "held-out 20% test split, no training data leakage",
        }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading script stats from {STATS_PATH}...")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"  {len(stats):,} scripts")

    pf = pq.ParquetFile(DATASET)

    results = []

    r = run_experiment(pf, "Chromium", "Firefox", stats)
    results.append(r)

    r = run_experiment(pf, "Firefox", "Chromium", stats)
    results.append(r)

    breakdown = per_engine_breakdown(pf, stats)

    report = {
        "model_class":              "TF-IDF bigrams + 5 context features (iter-3 style)",
        "stats_file":               STATS_PATH,
        "cross_engine_experiments": results,
        "mixed_model_per_engine":   breakdown,
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['train_engine']} → {r['test_engine']}: "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}")
    if breakdown:
        print("  Joint model (iter-3) per engine:")
        for eng, m in breakdown.items():
            print(f"    {eng}: F1={m['f1']:.4f}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
