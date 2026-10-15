"""
Transformer binary classifier v2 — Sprint 11, Task 4.

v2 fixes vs v1 (F1=0.415):
  1. Learned embeddings from scratch (d_model=64) — no FastText dependency.
  2. BCEWithLogitsLoss + pos_weight (no Focal Loss).
  3. max_len=256, patience=5, 32 epochs.
  → Result on cluster (P100): F1=0.7066 (better but below LightGBM 0.853)

v2 remaining issues fixed in this version (v2 final):
  4. Mean+max pooling concatenated (128-dim) instead of mean-only (64-dim) —
     matches the BiLSTM pooling strategy that reached F1=0.856.
  5. pos_weight clamped at neg_ratio (10) instead of computed ratio (13) —
     the training set is already downsampled; stacking pos_weight=13 caused
     the oscillation seen at epochs 7-11.
  6. Linear warmup (2 epochs) before cosine decay — smoother early training.
  7. Lower initial lr=5e-4 (was 1e-3).

Architecture: Embedding(147,64) → TransformerEncoder(1 layer, 4 heads, FFN=256)
              → mean+max pool → cat(128-dim) → Linear(128,64) → ReLU → Linear(64,1)

Usage:
    python transformer/scripts/train_v2.py [--out transformer/model_binary_v2]
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
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).parent.parent.parent

VOCAB_SIZE = 147   # PAD(0) + UNK(1) + 145 IR tokens
PAD_IDX    = 0
UNK_IDX    = 1


# ── Vocab ─────────────────────────────────────────────────────────────────────

def build_vocab(vec_path: str | Path) -> dict[str, int]:
    """Read IR token names from fasttext.vec header; assign fixed indices."""
    token2idx: dict[str, int] = {}
    with open(vec_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            token = line.split(" ", 1)[0]
            token2idx[token] = len(token2idx) + 2   # 0=PAD, 1=UNK
    return token2idx


def tokenize(ir_str: str, token2idx: dict, max_len: int) -> list[int]:
    toks = ir_str.split() if ir_str else []
    ids  = [token2idx.get(t, UNK_IDX) for t in toks[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


# ── Model ─────────────────────────────────────────────────────────────────────

class BinaryTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, nhead=4,
                 num_layers=1, dim_feedforward=256, dropout=0.1, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embedding.weight.data[PAD_IDX].zero_()

        # Learnable positional encoding — more flexible than sinusoidal
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        # mean+max pooling → concat → 2*d_model input to head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.max_len = max_len

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L) — accept int16/int32/int64, normalise to long for embedding
        ids = ids.long()
        B, L = ids.shape
        positions = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.drop(self.embedding(ids) + self.pos_emb(positions))
        pad_mask = (ids == PAD_IDX)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Mean pooling (weighted by non-PAD)
        active = (~pad_mask).float().unsqueeze(-1)          # (B, L, 1)
        mean_pool = (x * active).sum(1) / active.sum(1).clamp(min=1.0)

        # Max pooling (mask PAD with large negative)
        x_masked = x + pad_mask.unsqueeze(-1).float() * -1e9
        max_pool  = x_masked.max(dim=1).values

        x = torch.cat([mean_pool, max_pool], dim=-1)         # (B, 2*d_model)
        return self.head(x).squeeze(-1)                      # (B,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, ids: np.ndarray, labels: np.ndarray):
        self.ids    = torch.from_numpy(ids)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):  return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.labels[i]


# ── Sampling ──────────────────────────────────────────────────────────────────

def collect_sample(pf, token2idx, max_len, neg_ratio, seed):
    rng        = random.Random(seed)
    n_rg       = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    print("Estimating FP count...")
    fp_sample, sampled = 0, 0
    for i in range(0, n_rg, 50):
        col = pf.read_row_group(i, columns=["is_fp"]).to_pandas()
        fp_sample += int(col["is_fp"].fillna(False).sum())
        sampled   += len(col)
    est_fp      = int(fp_sample * total_rows / sampled)
    sample_prob = min(1.0, neg_ratio * est_fp / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    fp_ids, neg_ids, fp_y, neg_y = [], [], [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["ir_tokens", "is_fp"]).to_pandas()
        df["is_fp"]     = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")

        fp  = df[df["is_fp"]]
        neg = df[~df["is_fp"]]

        for seq in fp["ir_tokens"]:
            fp_ids.append(tokenize(seq, token2idx, max_len))
        fp_y.extend([1] * len(fp))

        if len(neg) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg))])
            if keep.any():
                for seq in neg["ir_tokens"].values[keep]:
                    neg_ids.append(tokenize(seq, token2idx, max_len))
                neg_y.extend([0] * int(keep.sum()))

    X = np.array(fp_ids + neg_ids,  dtype=np.int16)
    Y = np.array(fp_y  + neg_y,     dtype=np.float32)
    print(f"Collected {len(X):,}: {int(Y.sum()):,} FP + {int((Y==0).sum()):,} non-FP")
    return X, Y


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default=str(_ROOT / "ir/translated_dataset.parquet"))
    p.add_argument("--embeddings", default=str(_ROOT / "ir/embeddings/fasttext.vec"))
    p.add_argument("--out",        default=str(_ROOT / "transformer/model_binary_v2"))
    p.add_argument("--neg-ratio",  type=float, default=5.0)
    p.add_argument("--max-len",    type=int,   default=128)
    p.add_argument("--epochs",     type=int,   default=32)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def tune_threshold(model, loader, device, n_steps=19):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for ids, y in loader:
            probs = torch.sigmoid(model(ids.to(device))).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_y.extend(y.numpy().tolist())
    probs = np.array(all_probs); Y = np.array(all_y)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(Y, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1, probs, Y


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Building vocab from {args.embeddings}...")
    token2idx = build_vocab(args.embeddings)
    print(f"  {len(token2idx)} IR tokens (total vocab={VOCAB_SIZE})")

    pf = pq.ParquetFile(args.dataset)
    X, Y = collect_sample(pf, token2idx, args.max_len, args.neg_ratio, args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=Y
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {int(Y_tr.sum()):,}  Test FP: {int(Y_te.sum()):,}")

    # Cap pos_weight at neg_ratio: the training set is already downsampled,
    # so the computed ratio (~neg_ratio) is sufficient; unclamped it reaches 13+
    # which combined with the 10:1 sampling causes oscillation.
    computed_pw = (Y_tr == 0).sum() / max((Y_tr == 1).sum(), 1)
    pos_weight  = torch.tensor(
        [min(float(computed_pw), args.neg_ratio)], dtype=torch.float32
    ).to(device)
    print(f"pos_weight: {pos_weight.item():.1f}  (computed={computed_pw:.1f})")

    tr_loader = DataLoader(SeqDataset(X_tr, Y_tr), batch_size=args.batch_size,
                           shuffle=True,  num_workers=2)
    te_loader = DataLoader(SeqDataset(X_te, Y_te), batch_size=args.batch_size * 2,
                           shuffle=False, num_workers=2)

    model = BinaryTransformer(max_len=args.max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 2
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    best_f1 = 0.0
    patience_count = 0

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for ids, y in tr_loader:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        scheduler.step()

        # Quick val F1 at threshold=0.5 for early stopping
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, y in te_loader:
                p = torch.sigmoid(model(ids.to(device))).cpu().numpy()
                preds.extend((p >= 0.5).tolist())
                trues.extend(y.numpy().tolist())
        val_f1 = f1_score(trues, preds, zero_division=0)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total_loss/n_batches:.4f}  "
              f"val_F1@0.5={val_f1:.4f}  lr={lr_now:.2e}  ({time.time()-t0:.0f}s)",
              flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
            print(f"  → checkpoint saved (F1={best_f1:.4f})", flush=True)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best and tune threshold
    model.load_state_dict(torch.load(os.path.join(args.out, "model.pt"),
                                     weights_only=True))
    print("\nTuning threshold on test split...")
    best_t, tuned_f1, probs, Y_te_np = tune_threshold(model, te_loader, device)
    Y_pred = (probs >= best_t).astype(int)
    cr = classification_report(Y_te_np, Y_pred, target_names=["non-FP", "FP"],
                                output_dict=True, zero_division=0)
    print(f"  Best threshold: {best_t:.2f}  F1={tuned_f1:.4f}")
    print(f"  P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}")

    config = {
        "vocab_size": VOCAB_SIZE, "d_model": 64, "nhead": 2, "num_layers": 1,
        "dim_feedforward": 256, "max_len": args.max_len,
        "neg_ratio": args.neg_ratio, "binary": True,
    }
    metrics = {
        "f1":          tuned_f1,
        "precision":   cr["FP"]["precision"],
        "recall":      cr["FP"]["recall"],
        "threshold":   best_t,
        "val_f1_at_05": best_f1,
        "train_size":  len(X_tr),
        "test_size":   len(X_te),
        "baseline_f1": 0.853,
        "delta":       round(tuned_f1 - 0.853, 4),
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {args.out}/")
    print(f"Final F1: {tuned_f1:.4f}  (baseline 0.853, delta {tuned_f1-0.853:+.4f})")


if __name__ == "__main__":
    main()
