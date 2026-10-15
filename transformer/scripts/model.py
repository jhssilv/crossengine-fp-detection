"""
Fingerprint Transformer — Sprint 8.

Architecture follows ByteDefender (2025) adapted for:
  - dim=128 (matches FastText IR embeddings, vs ByteDefender's 256)
  - 7-label multilabel sigmoid output (vs ByteDefender's binary FP/non-FP)
  - Pretrained embedding initialization from FastText skipgram
  - Pre-norm TransformerEncoderLayer (more stable than post-norm for shallow nets)
"""

import math
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class FingerprintTransformer(nn.Module):
    """
    Transformer encoder for multilabel bytecode fingerprint classification.

    Follows ByteDefender's architecture:
      Embedding → Positional Encoding → TransformerEncoder (1 layer, 4 heads, FFN=512)
      → Global Average Pooling → Dense(ReLU, Dropout) × 2 → Linear(7)

    Outputs raw logits; use BCEWithLogitsLoss during training.
    """

    LABEL_NAMES = [
        "label_canvas", "label_audio", "label_webrtc", "label_font",
        "label_webgl", "label_screen", "label_navigator",
    ]

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_labels: int = 7,
        max_len: int = 512,
        pretrained_embeddings: np.ndarray | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        # Token embedding — initialized from FastText if provided
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings[:vocab_size])
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad_(False)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for single-layer nets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head (ByteDefender: 2 Dense + ReLU + Dropout → output)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels),
        )

    def forward(
        self,
        token_ids: torch.Tensor,          # (batch, seq)
        padding_mask: torch.Tensor | None = None,  # (batch, seq) bool, True = ignore
    ) -> torch.Tensor:
        x = self.embedding(token_ids)     # (batch, seq, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Global average pooling — exclude PAD positions
        if padding_mask is not None:
            active = (~padding_mask).float().unsqueeze(-1)  # (batch, seq, 1)
            x = (x * active).sum(1) / active.sum(1).clamp(min=1.0)
        else:
            x = x.mean(1)

        return self.head(x)  # (batch, num_labels) raw logits
