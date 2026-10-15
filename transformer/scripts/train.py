"""
Sprint 8 — Transformer multilabel classifier.

Pipeline:
  1. Stream translated_dataset.parquet (same neg_ratio sampling as RF baseline)
  2. Build token-id arrays + label matrix in memory
  3. Train/val split (stratified on is_fp)
  4. PyTorch Dataset + DataLoader (pin_memory, num_workers)
  5. Focal loss with per-label pos_weight for extreme class imbalance
  6. Adam + linear warmup + CosineAnnealingLR, mixed precision (AMP) on GPU
  7. Early stopping on val macro-F1 (patience=3)
  8. Save model.pt, config.json, metrics.json

Usage:
    python transformer/scripts/train.py [options]

Key options:
    --dataset        path to translated_dataset.parquet
    --embeddings     path to ir/embeddings/fasttext.vec
    --out            output directory (default: transformer/model)
    --neg-ratio      non-FP rows per FP row (default: 5)
    --max-len        truncation length (default: 512)
    --num-layers     Transformer encoder layers (default: 2)
    --epochs         default: 16
    --batch-size     default: 128
    --lr             default: 3e-4
    --warmup-epochs  linear LR warmup before cosine decay (default: 2)
    --gamma          focal loss focusing parameter (default: 2.0)
    --patience       early stopping patience (default: 3)
    --freeze-emb     freeze FastText embeddings during training
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from vocab import load_vocab_and_embeddings, tokenize, PAD_IDX
from model import FingerprintTransformer

LABEL_COLS = [
    "label_canvas", "label_audio", "label_webrtc", "label_font",
    "label_webgl", "label_screen", "label_navigator",
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class BytecodeDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, labels: np.ndarray):
        # token_ids: (N, max_len) int32
        # labels:    (N, 7)       int8
        self.token_ids = torch.from_numpy(token_ids)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


# ── Focal loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss for multilabel classification.

    Applies per-label pos_weight (same as BCEWithLogitsLoss) combined with
    focal modulation (1-p_t)^gamma to down-weight easy examples. This helps
    with extreme imbalance where weighted BCE still wastes capacity on the
    flood of easy negatives.
    """
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        focal = (1 - p_t) ** self.gamma
        alpha = self.pos_weight * targets + (1 - targets)
        return (focal * alpha * bce).mean()


# ── LR scheduler: linear warmup → cosine decay ───────────────────────────────

def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-5):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ── Streaming sampler (same strategy as train_rf.py) ─────────────────────────

def collect_sample(
    pf: pq.ParquetFile,
    token2idx: dict,
    max_len: int,
    neg_ratio: float,
    seed: int,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    # Estimate n_fp from 1-in-50 row group sample
    print("Estimating FP count...", flush=True)
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

    label_cols = LABEL_COLS if not binary else []
    cols = ["ir_tokens", "is_fp"] + label_cols
    fp_ids, fp_lbls = [], []
    neg_ids, neg_lbls = [], []
    t0 = time.time()

    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=cols).to_pandas()
        df["is_fp"] = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")

        fp_df = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]

        if len(fp_df):
            for seq in fp_df["ir_tokens"]:
                fp_ids.append(tokenize(seq, token2idx, max_len))
            if binary:
                fp_lbls.append(np.ones((len(fp_df), 1), dtype=np.int8))
            else:
                fp_lbls.append(fp_df[LABEL_COLS].values.astype(np.int8))

        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                for seq in neg_df["ir_tokens"].values[keep]:
                    neg_ids.append(tokenize(seq, token2idx, max_len))
                if binary:
                    neg_lbls.append(np.zeros((int(keep.sum()), 1), dtype=np.int8))
                else:
                    neg_lbls.append(neg_df[LABEL_COLS].values[keep].astype(np.int8))

    print(f"Collected {len(fp_ids):,} FP + {len(neg_ids):,} non-FP rows", flush=True)

    n_labels = 1 if binary else len(LABEL_COLS)
    all_ids = np.array(fp_ids + neg_ids, dtype=np.int32)
    Y_fp = np.concatenate(fp_lbls, axis=0) if fp_lbls else np.empty((0, n_labels), np.int8)
    Y_neg = np.concatenate(neg_lbls, axis=0) if neg_lbls else np.empty((0, n_labels), np.int8)
    all_labels = np.concatenate([Y_fp, Y_neg], axis=0)

    return all_ids, all_labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, threshold=0.5, binary=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for token_ids, labels in loader:
            token_ids = token_ids.to(device)
            padding_mask = (token_ids == PAD_IDX)
            logits = model(token_ids, padding_mask)
            preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = np.concatenate(all_labels, axis=0)
    if binary:
        f1 = f1_score(Y_true.ravel(), Y_pred.ravel(), zero_division=0)
        return f1, f1, np.array([f1])
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    return macro, micro, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ir/translated_dataset.parquet")
    p.add_argument("--embeddings", default="ir/embeddings/fasttext.vec")
    p.add_argument("--out", default="transformer/model")
    p.add_argument("--neg-ratio", type=float, default=10.0)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--freeze-emb", action="store_true",
                   help="Freeze FastText embeddings (train head only first)")
    p.add_argument("--binary", action="store_true",
                   help="Binary classifier (is_fp) instead of 7-label multilabel")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # ── Vocab + embeddings ────────────────────────────────────────────────────
    print(f"\nLoading vocab from {args.embeddings}...", flush=True)
    token2idx, emb_matrix = load_vocab_and_embeddings(args.embeddings)
    vocab_size, d_model = emb_matrix.shape
    print(f"  Vocab size: {vocab_size}  Embedding dim: {d_model}")

    # ── Stream dataset ────────────────────────────────────────────────────────
    mode = "binary" if args.binary else "multilabel"
    print(f"\nMode: {mode}", flush=True)
    print(f"Streaming {args.dataset}...", flush=True)
    pf = pq.ParquetFile(args.dataset)
    X, Y = collect_sample(pf, token2idx, args.max_len, args.neg_ratio, args.seed, args.binary)

    # ── Train/val split (stratify on is_fp) ───────────────────────────────────
    is_fp = Y.any(axis=1)
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=is_fp
    )
    print(f"Train: {len(X_tr):,}  Val: {len(X_val):,}", flush=True)

    # ── Class weights (pos_weight = n_neg / n_pos per label) ─────────────────
    n_tr = len(Y_tr)
    pos_counts = Y_tr.sum(axis=0).astype(float)
    neg_counts = n_tr - pos_counts
    pos_weight = torch.tensor(
        np.where(pos_counts > 0, neg_counts / pos_counts, 1.0),
        dtype=torch.float32,
        device=device,
    )
    if args.binary:
        print(f"\nPos weight (is_fp): {pos_weight[0]:.1f}")
    else:
        print("\nPos weights per label:")
        for name, w in zip(LABEL_COLS, pos_weight.tolist()):
            print(f"  {name:20s}: {w:.1f}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    tr_set = BytecodeDataset(X_tr, Y_tr)
    val_set = BytecodeDataset(X_val, Y_val)

    tr_loader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_labels = 1 if args.binary else len(LABEL_COLS)
    model = FingerprintTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=args.num_layers,
        dim_feedforward=512,
        dropout=0.1,
        num_labels=num_labels,
        max_len=args.max_len,
        pretrained_embeddings=emb_matrix,
        freeze_embeddings=args.freeze_emb,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}", flush=True)

    criterion = FocalLoss(pos_weight=pos_weight, gamma=args.gamma)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_macro_f1 = 0.0
    patience_count = 0
    history = []

    print("\nStarting training...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_batches = 0

        for token_ids, labels in tr_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            padding_mask = (token_ids == PAD_IDX)

            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(token_ids, padding_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        macro, micro, per_class = evaluate(model, val_loader, device, binary=args.binary)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:2d}/{args.epochs}  loss={epoch_loss/n_batches:.4f}  "
            f"val macro-F1={macro:.4f}  micro-F1={micro:.4f}  "
            f"lr={lr_now:.2e}  ({elapsed:.0f}s)",
            flush=True,
        )
        if not args.binary:
            for name, f1 in zip(LABEL_COLS, per_class.tolist()):
                print(f"    {name:20s}  F1={f1:.3f}", flush=True)

        entry = {
            "epoch": epoch,
            "loss": epoch_loss / n_batches,
            "val_macro_f1": macro,
            "val_micro_f1": micro,
        }
        if not args.binary:
            for name, f1 in zip(LABEL_COLS, per_class.tolist()):
                entry[name] = f1
        history.append(entry)

        # Save best checkpoint
        if macro > best_macro_f1:
            best_macro_f1 = macro
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
            print(f"  → New best macro-F1={best_macro_f1:.4f} — checkpoint saved", flush=True)
        else:
            patience_count += 1
            print(f"  → No improvement ({patience_count}/{args.patience})", flush=True)
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    # ── Save config + metrics ─────────────────────────────────────────────────
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": 4,
        "num_layers": args.num_layers,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "num_labels": num_labels,
        "max_len": args.max_len,
        "neg_ratio": args.neg_ratio,
        "binary": args.binary,
        "epochs_trained": epoch,
        "best_val_macro_f1": best_macro_f1,
        "label_names": ["is_fp"] if args.binary else LABEL_COLS,
        "train_size": len(X_tr),
        "val_size": len(X_val),
    }

    best = max(history, key=lambda h: h["val_macro_f1"])
    metrics = {
        "macro_f1": best["val_macro_f1"],
        "micro_f1": best["val_micro_f1"],
        "best_epoch": best["epoch"],
        "train_size": len(X_tr),
        "val_size": len(X_val),
        "neg_ratio": args.neg_ratio,
        "history": history,
    }
    if not args.binary:
        for name in LABEL_COLS:
            metrics[name] = {"f1-score": best[name]}

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Best val macro-F1={best_macro_f1:.4f}")
    print(f"Saved → {args.out}/model.pt, config.json, metrics.json")


if __name__ == "__main__":
    main()
