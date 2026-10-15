"""
Train Word2Vec and/or FastText embeddings on the IR token corpus.

Steps:
  1. Stream ir/translated_dataset.parquet → write ir/corpus.txt (one function per line)
  2. Train model(s) via fasttext
  3. Save .bin model + .vec word vectors to ir/embeddings/

Usage:
  python ir/train_embeddings.py                      # both models, full corpus
  python ir/train_embeddings.py --model word2vec     # Word2Vec only
  python ir/train_embeddings.py --sample 1000000     # 1 M rows (quick test)
  python ir/train_embeddings.py --dim 256 --epochs 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import fasttext
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
TRANSLATED_PATH = ROOT / "ir/translated_dataset.parquet"
CORPUS_PATH = ROOT / "ir/corpus.txt"
EMBEDDINGS_DIR = ROOT / "ir/embeddings"

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — write corpus
# ---------------------------------------------------------------------------

def write_corpus(
    sample: int | None,
    min_tokens: int = 2,
    chunk_size: int = 512,
) -> int:
    """
    Stream translated_dataset.parquet and write one IR token sequence per line
    to corpus.txt.

    - Sequences shorter than min_tokens are skipped (CBOW has no valid context).
    - Sequences longer than chunk_size are split into non-overlapping chunks.
      No tokens are discarded: a 5830-token function becomes ceil(5830/512)
      training sentences. Word2Vec only ever sees a local context window anyway,
      so chunking is semantically equivalent to the full sequence — but
      truncation would silently drop fingerprinting opcodes from long functions.

    Returns number of lines written to corpus.txt.
    """
    pf = pq.ParquetFile(TRANSLATED_PATH)
    total_rows = pf.metadata.num_rows
    total_rgs = pf.metadata.num_row_groups

    log.info(
        "Writing corpus — source: %s (%d rows, %d row groups, min=%d chunk=%d tokens)",
        TRANSLATED_PATH.name, total_rows, total_rgs, min_tokens, chunk_size,
    )
    if sample:
        log.info("Sample mode: will stop after %d source rows", sample)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    src_rows = 0
    rows_skipped = 0
    rows_chunked = 0
    lines_written = 0
    rg_idx = 0
    run_start = time.perf_counter()

    with open(CORPUS_PATH, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(columns=["ir_tokens"]):
            ir_col = batch.column("ir_tokens").to_pylist()

            for seq in ir_col:
                tokens = seq.split(" ")
                if len(tokens) < min_tokens:
                    rows_skipped += 1
                else:
                    if len(tokens) > chunk_size:
                        rows_chunked += 1
                    for start in range(0, len(tokens), chunk_size):
                        fout.write(" ".join(tokens[start:start + chunk_size]) + "\n")
                        lines_written += 1
                src_rows += 1
                if sample and src_rows >= sample:
                    break

            rg_idx += 1
            elapsed = time.perf_counter() - run_start
            pct = 100 * src_rows / total_rows
            eta = (elapsed / src_rows) * (total_rows - src_rows) if src_rows else 0
            log.info(
                "Corpus RG %d/%d | src %d/%d (%.1f%%) | lines %d | skipped %d | chunked %d | elapsed %.0fs | ETA %.0fs",
                rg_idx, total_rgs, src_rows, total_rows, pct,
                lines_written, rows_skipped, rows_chunked, elapsed, eta,
            )

            if sample and src_rows >= sample:
                break

    elapsed = time.perf_counter() - run_start
    size_mb = CORPUS_PATH.stat().st_size / 1024 / 1024
    log.info(
        "Corpus written — %d src rows → %d lines | skipped %d short | chunked %d long | %.1f MB | %.0fs",
        src_rows, lines_written, rows_skipped, rows_chunked, size_mb, elapsed,
    )
    return lines_written


# ---------------------------------------------------------------------------
# Step 2 — train
# ---------------------------------------------------------------------------

def train(
    model_type: str,
    dim: int,
    epochs: int,
    window: int,
    workers: int,
) -> None:
    """Train a single model and save .bin + .vec to ir/embeddings/."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBEDDINGS_DIR / model_type  # fasttext appends .bin / .vec

    # Word2Vec: disable subword (minn=maxn=0) so tokens are treated atomically.
    # FastText: keep defaults (minn=3, maxn=6) for subword character n-grams.
    subword = model_type == "fasttext"
    algo = "skipgram" if model_type == "fasttext" else "cbow"
    minn = 3 if subword else 0
    maxn = 6 if subword else 0

    log.info(
        "Training %s — algo=%s dim=%d window=%d epochs=%d workers=%d subword=%s",
        model_type, algo, dim, window, epochs, workers, subword,
    )
    t0 = time.perf_counter()

    model = fasttext.train_unsupervised(
        str(CORPUS_PATH),
        model=algo,
        dim=dim,
        ws=window,
        epoch=epochs,
        minCount=1,       # keep all 138 IR tokens regardless of frequency
        minn=minn,
        maxn=maxn,
        thread=workers,
        verbose=2,
        # t=0.0: disable frequency-based subsampling. The default (t=1e-4)
        # aggressively drops high-frequency tokens; with only ~138 IR types
        # every token qualifies as "high-frequency", gutting the training signal.
        t=0.0,
    )

    elapsed = time.perf_counter() - t0
    log.info("Training done in %.0fs — vocab size: %d", elapsed, len(model.words))

    bin_path = str(out_path) + ".bin"
    model.save_model(bin_path)
    log.info("Model saved → %s", bin_path)

    vec_path = str(out_path) + ".vec"
    _save_vec(model, vec_path)
    log.info("Vectors saved → %s", vec_path)


def _save_vec(model: fasttext.FastText._FastText, path: str) -> None:
    """Write a standard .vec (word2vec text) file from a fasttext model."""
    words = model.words
    dim = model.get_dimension()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for word in words:
            vec = " ".join(f"{v:.6f}" for v in model.get_word_vector(word))
            f.write(f"{word} {vec}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["word2vec", "fasttext", "both"], default="both",
                   help="Which model(s) to train (default: both)")
    p.add_argument("--dim", type=int, default=128, help="Embedding dimension (default: 128)")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    p.add_argument("--window", type=int, default=5, help="Context window size (default: 5)")
    p.add_argument("--workers", type=int, default=4, help="CPU threads for fasttext (default: 4)")
    p.add_argument("--sample", type=int, default=None,
                   help="Limit corpus to N rows (default: all). Use for quick tests.")
    p.add_argument("--min-tokens", type=int, default=2,
                   help="Skip sequences shorter than N tokens (default: 2).")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Split sequences longer than N tokens into chunks (default: 512). "
                        "No tokens are discarded.")
    p.add_argument("--skip-corpus", action="store_true",
                   help="Reuse existing corpus.txt instead of rewriting it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()

    log.info("=== train_embeddings.py start ===")
    log.info(
        "Config — model=%s dim=%d epochs=%d window=%d workers=%d sample=%s",
        args.model, args.dim, args.epochs, args.window, args.workers,
        args.sample if args.sample else "all",
    )

    # Step 1
    if args.skip_corpus and CORPUS_PATH.exists():
        size_mb = CORPUS_PATH.stat().st_size / 1024 / 1024
        log.info("Reusing existing corpus: %s (%.1f MB)", CORPUS_PATH, size_mb)
    else:
        write_corpus(args.sample, args.min_tokens, args.chunk_size)

    # Step 2
    models_to_train = (
        ["word2vec", "fasttext"] if args.model == "both" else [args.model]
    )
    for m in models_to_train:
        train(m, args.dim, args.epochs, args.window, args.workers)

    total = time.perf_counter() - run_start
    log.info("=== train_embeddings.py done in %.0fs ===", total)


if __name__ == "__main__":
    main()
