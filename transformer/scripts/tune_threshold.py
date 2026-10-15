"""
Per-label decision threshold tuning.

Recreates the exact val split used during training (same seed + neg_ratio from
config.json), runs inference with the saved model.pt, and grid-searches the
decision threshold for each label to maximise F1. Saves thresholds.json next
to model.pt so inference scripts can load it.

Usage:
    python tune_threshold.py \
        --model-dir  path/to/model_dir \
        --dataset    path/to/translated_dataset.parquet \
        --embeddings path/to/fasttext.vec
"""

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import FingerprintTransformer
from train import LABEL_COLS, BytecodeDataset, collect_sample
from vocab import PAD_IDX, load_vocab_and_embeddings


def collect_probs(model, val_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for token_ids, labels in val_loader:
            token_ids = token_ids.to(device)
            padding_mask = (token_ids == PAD_IDX)
            logits = model(token_ids, padding_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def search_thresholds(probs, labels, label_names, n_steps=19):
    thresholds = {}
    print(f"\n  {'Label':<22} {'@0.50':>8} {'Best F1':>8} {'Thresh':>8}")
    print("  " + "-" * 50)
    for i, name in enumerate(label_names):
        default_f1 = f1_score(labels[:, i], probs[:, i] >= 0.5, zero_division=0)
        best_f1, best_t = default_f1, 0.5
        for t in np.linspace(0.05, 0.95, n_steps):
            f = f1_score(labels[:, i], probs[:, i] >= t, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds[name] = best_t
        delta = f"(+{best_f1 - default_f1:.3f})" if best_f1 - default_f1 > 1e-4 else ""
        print(f"  {name:<22} {default_f1:>8.3f} {best_f1:>8.3f} {best_t:>8.2f}  {delta}")

    default_macro = f1_score(labels, probs >= 0.5, average="macro", zero_division=0)
    tuned = np.stack(
        [(probs[:, i] >= thresholds[n]).astype(int) for i, n in enumerate(label_names)], axis=1
    )
    tuned_macro = f1_score(labels, tuned, average="macro", zero_division=0)
    print(f"\n  macro-F1: {default_macro:.4f} → {tuned_macro:.4f}  (Δ={tuned_macro - default_macro:+.4f})")
    return thresholds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading vocab from {args.embeddings}...")
    token2idx, emb_matrix = load_vocab_and_embeddings(args.embeddings)

    print("Streaming dataset to reconstruct val split...")
    pf = pq.ParquetFile(args.dataset)
    X, Y = collect_sample(
        pf, token2idx,
        max_len=config["max_len"],
        neg_ratio=config["neg_ratio"],
        seed=42,
        binary=config.get("binary", False),
    )
    is_fp = Y.any(axis=1)
    _, X_val, _, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=is_fp
    )
    print(f"Val set: {len(X_val):,} rows")

    batch_size = 512 if device.type == "cuda" else 64
    val_loader = DataLoader(
        BytecodeDataset(X_val, Y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FingerprintTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=0.0,
        num_labels=config["num_labels"],
        max_len=config["max_len"],
        pretrained_embeddings=emb_matrix,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device)
    )

    print("Running inference...")
    probs, labels = collect_probs(model, val_loader, device)

    label_names = config.get("label_names", LABEL_COLS)
    print("Searching thresholds...")
    thresholds = search_thresholds(probs, labels, label_names)

    out_path = os.path.join(args.model_dir, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
