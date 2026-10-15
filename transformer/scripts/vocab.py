"""
Vocabulary utilities for the IR token transformer.

Reads the FastText .vec file to build the canonical token→index mapping and
provides the pretrained embedding matrix for weight initialization.

Special indices:
  0 = <PAD>   (used for padding, embedding row is zeroed / not updated)
  1 = <UNK>   (unmapped tokens — should be rare on IR sequences)
  2..N = IR tokens in FastText vocab order
"""

import numpy as np
from pathlib import Path

PAD_IDX = 0
UNK_IDX = 1


def load_vocab_and_embeddings(vec_path: str | Path):
    """Parse fasttext.vec and return (token2idx, embedding_matrix).

    embedding_matrix shape: (vocab_size, dim)
    Rows 0 (<PAD>) and 1 (<UNK>) are zero-initialized.
    """
    vec_path = Path(vec_path)
    token2idx: dict[str, int] = {}
    vectors: list[np.ndarray] = []

    with open(vec_path, encoding="utf-8") as f:
        header = f.readline().split()
        n_words, dim = int(header[0]), int(header[1])

        # Reserve slots 0 and 1
        pad_vec = np.zeros(dim, dtype=np.float32)
        unk_vec = np.zeros(dim, dtype=np.float32)
        vectors.append(pad_vec)  # idx 0 = PAD
        vectors.append(unk_vec)  # idx 1 = UNK

        for line in f:
            parts = line.rstrip().split(" ")
            token = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) != dim:
                continue
            token2idx[token] = len(vectors)
            vectors.append(vec)

    embedding_matrix = np.stack(vectors, axis=0)  # (vocab_size, dim)
    return token2idx, embedding_matrix


def tokenize(ir_tokens_str: str, token2idx: dict[str, int],
             max_len: int = 512) -> list[int]:
    """Convert a space-separated IR token string to a padded index list."""
    tokens = ir_tokens_str.split() if ir_tokens_str else []
    ids = [token2idx.get(t, UNK_IDX) for t in tokens[:max_len]]
    # Pad to max_len
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids
