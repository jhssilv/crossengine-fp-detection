"""
Binary fingerprinting classifier: BiLSTM trained end-to-end on IR token sequences.

Unlike the TF-IDF models (bag-of-n-grams), this model processes the actual
token sequence using a bidirectional LSTM, capturing long-range order information.

Architecture:
  Embedding(vocab_size, emb_dim=64)
  → BiLSTM(64, hidden=256, num_layers=2, dropout=0.3)
  → mean + max pooling over time
  → Linear(512, 1) + sigmoid

Loss: BCEWithLogitsLoss with pos_weight to handle class imbalance.

Usage:
    python ir/scripts/train_bilstm.py [--neg-ratio 10] [--epochs 10] [--out ir/bilstm_model]
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
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

_IR_ROOT = Path(__file__).parent.parent
DATASET  = str(_IR_ROOT / "translated_dataset.parquet")

PAD_ID = 0   # reserved for padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neg-ratio",   type=float, default=10.0)
    p.add_argument("--max-len",     type=int,   default=512)
    p.add_argument("--emb-dim",     type=int,   default=64)
    p.add_argument("--hidden-dim",  type=int,   default=256)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--out",         default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--dataset",     default=DATASET)
    return p.parse_args()


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(sequences):
    """Build token → id mapping from a list of space-separated token strings."""
    tokens = set()
    for seq in sequences:
        tokens.update(seq.split())
    vocab = {tok: idx + 1 for idx, tok in enumerate(sorted(tokens))}  # 0=PAD
    return vocab


def encode(seq, vocab, max_len):
    ids = [vocab.get(t, PAD_ID) for t in seq.split()]
    ids = ids[:max_len]
    pad = max_len - len(ids)
    return ids + [PAD_ID] * pad


# ── Dataset ──────────────────────────────────────────────────────────────────

class FPDataset(torch.utils.data.Dataset):
    def __init__(self, X_ids, Y):
        self.X = torch.tensor(X_ids, dtype=torch.long)
        self.Y = torch.tensor(Y,     dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_id=PAD_ID):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, 1)  # *4: bidirectional × (mean+max)

    def forward(self, x):
        mask = (x != PAD_ID).float().unsqueeze(-1)          # (B, L, 1)
        emb  = self.dropout(self.embedding(x))               # (B, L, E)
        out, _ = self.lstm(emb)                              # (B, L, 2H)

        # mean pooling (ignoring padding)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_pool = (out * mask).sum(dim=1) / lengths        # (B, 2H)

        # max pooling (set padding positions to -inf before max)
        out_masked = out - (1 - mask) * 1e9
        max_pool = out_masked.max(dim=1).values              # (B, 2H)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)    # (B, 4H)
        return self.fc(self.dropout(pooled)).squeeze(-1)     # (B,)


# ── Sampling ──────────────────────────────────────────────────────────────────

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
    target_neg = int(neg_ratio * est_fp)
    sample_prob = min(1.0, target_neg / max(total_rows - est_fp, 1))
    print(f"  est FP={est_fp:,}  sample_prob={sample_prob:.4f}")

    fp_rows, neg_rows = [], []
    t0 = time.time()
    for i in range(n_rg):
        if i % 100 == 0:
            print(f"  row group {i}/{n_rg}  ({time.time()-t0:.0f}s)", flush=True)
        df = pf.read_row_group(i, columns=["ir_tokens", "is_fp"]).to_pandas()
        df["is_fp"]    = df["is_fp"].fillna(False).astype(bool)
        df["ir_tokens"] = df["ir_tokens"].fillna("")

        fp_df  = df[df["is_fp"]]
        neg_df = df[~df["is_fp"]]

        if len(fp_df):
            fp_rows.extend(fp_df["ir_tokens"].tolist())
        if len(neg_df) and sample_prob > 0:
            keep = np.array([rng.random() < sample_prob for _ in range(len(neg_df))])
            if keep.any():
                neg_rows.extend(neg_df["ir_tokens"].values[keep].tolist())

    X = fp_rows + neg_rows
    Y = np.array([1] * len(fp_rows) + [0] * len(neg_rows), dtype=np.float32)
    print(f"Collected {len(X):,}: {len(fp_rows):,} FP + {len(neg_rows):,} non-FP")
    return X, Y


# ── Training ──────────────────────────────────────────────────────────────────

def tune_threshold(model, loader, device, n_steps=19):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, n_steps):
        f = f1_score(labels, probs >= t, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    default_f1 = f1_score(labels, probs >= 0.5, zero_division=0)
    print(f"  Threshold: @0.50={default_f1:.4f}  best={best_f1:.4f} @t={best_t:.2f}")
    return best_t, best_f1, probs, labels


def main():
    args = parse_args()
    if args.out is None:
        args.out = str(_IR_ROOT / "bilstm_model")
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pf = pq.ParquetFile(args.dataset)
    X, Y = collect_sample(pf, args.neg_ratio, args.seed)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=args.seed, stratify=Y.astype(int)
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
    print(f"Train FP: {int(Y_tr.sum()):,}  Test FP: {int(Y_te.sum()):,}")

    print("Building vocabulary...")
    vocab = build_vocab(X_tr)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    print(f"Encoding sequences (max_len={args.max_len})...")
    t0 = time.time()

    def encode_batch(seqs, vocab, max_len):
        out = np.zeros((len(seqs), max_len), dtype=np.int32)
        for i, s in enumerate(seqs):
            ids = [vocab.get(t, PAD_ID) for t in s.split()][:max_len]
            out[i, :len(ids)] = ids
        return out

    X_tr_ids = encode_batch(X_tr, vocab, args.max_len)
    X_te_ids = encode_batch(X_te, vocab, args.max_len)
    print(f"  Done in {time.time()-t0:.1f}s")

    tr_ds = FPDataset(X_tr_ids, Y_tr)
    te_ds = FPDataset(X_te_ids, Y_te)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size,
                                            shuffle=True,  num_workers=0)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=args.batch_size,
                                            shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = torch.tensor([args.neg_ratio], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        _, val_f1, _, _ = tune_threshold(model, te_loader, device)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total_loss/len(tr_loader):.4f}  "
              f"val_F1={val_f1:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val F1: {best_f1:.4f}")
    model.load_state_dict(best_state)
    best_t, best_f1, probs, labels = tune_threshold(model, te_loader, device)
    Y_pred = (probs >= best_t).astype(int)
    cr = classification_report(labels, Y_pred, target_names=["non-FP", "FP"],
                               output_dict=True, zero_division=0)
    print(f"Final  P={cr['FP']['precision']:.4f}  R={cr['FP']['recall']:.4f}  F1={best_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    config = {k: v for k, v in vars(args).items()}
    config["vocab_size"] = vocab_size
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "f1": best_f1,
        "precision": cr["FP"]["precision"],
        "recall": cr["FP"]["recall"],
        "threshold": best_t,
        "train_size": len(X_tr),
        "test_size": len(X_te),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {args.out}/")


if __name__ == "__main__":
    main()
