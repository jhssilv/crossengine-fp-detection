"""
Translate labeled_dataset_clean.parquet bytecode strings to IR token sequences.

Reads one row group at a time to stay within memory bounds (80M rows / 401 groups).
Writes ir/translated_dataset.parquet with bytecodes replaced by ir_tokens.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = ROOT / "data/labeled_dataset_clean.parquet"
EQDICT_PATH = ROOT / "ir/equivalence_dict.json"
DEFAULT_OUTPUT_PATH = ROOT / "ir/translated_dataset.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

UNK = "<UNK>"


def build_lookup(eq: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (v8_map, sm_map) plain dicts for fast per-token lookup."""
    return eq["v8_to_ir"], eq["sm_to_ir"]


def translate_sequence(bytecodes: str, mapping: dict[str, str]) -> str:
    """Comma-separated opcodes → space-separated IR tokens."""
    tokens = bytecodes.split(",")
    return " ".join(mapping.get(t, UNK) for t in tokens)


def translate_batch(
    batch: pa.RecordBatch,
    v8_map: dict[str, str],
    sm_map: dict[str, str],
) -> tuple[pa.RecordBatch, int]:
    """
    Translate all bytecode strings in a RecordBatch.
    Returns the new batch (bytecodes replaced by ir_tokens) and the unk count.
    """
    browsers: list[str] = batch.column("browser").to_pylist()
    bytecodes: list[str] = batch.column("bytecodes").to_pylist()

    ir_tokens: list[str] = []
    unk_count = 0

    for browser, bc in zip(browsers, bytecodes):
        mapping = sm_map if browser == "Firefox" else v8_map
        translated = translate_sequence(bc, mapping)
        unk_count += translated.count(UNK)
        ir_tokens.append(translated)

    # Drop bytecodes, add ir_tokens
    idx = batch.schema.get_field_index("bytecodes")
    new_batch = batch.remove_column(idx)
    ir_col = pa.array(ir_tokens, type=pa.string())
    new_batch = new_batch.append_column(
        pa.field("ir_tokens", pa.string()), ir_col
    )

    return new_batch, unk_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH),
                   help="Output parquet path (default: ir/translated_dataset.parquet)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_PATH = Path(args.output)
    log.info("Loading equivalence dict from %s", EQDICT_PATH)
    with open(EQDICT_PATH) as f:
        eq = json.load(f)
    v8_map, sm_map = build_lookup(eq)
    log.info(
        "Mappings loaded — V8: %d entries, SM: %d entries, IR tokens: %d",
        len(v8_map), len(sm_map), len(eq["ir_tokens"]),
    )

    pf = pq.ParquetFile(INPUT_PATH)
    total_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    log.info(
        "Input: %s — %d rows, %d row groups",
        INPUT_PATH.name, total_rows, total_row_groups,
    )
    log.info("Output: %s", OUTPUT_PATH)

    writer: pq.ParquetWriter | None = None
    rows_done = 0
    total_unks = 0
    run_start = time.perf_counter()

    try:
        for rg_idx, batch in enumerate(pf.iter_batches()):
            batch_start = time.perf_counter()

            new_batch, unk_count = translate_batch(batch, v8_map, sm_map)
            total_unks += unk_count

            # Initialise writer from the schema of the first translated batch
            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_PATH,
                    new_batch.schema,
                    compression="zstd",
                )

            writer.write_batch(new_batch)
            rows_done += len(batch)

            elapsed = time.perf_counter() - run_start
            batch_s = time.perf_counter() - batch_start
            pct = 100 * rows_done / total_rows
            eta = (elapsed / rows_done) * (total_rows - rows_done) if rows_done else 0

            log.info(
                "RG %d/%d | rows %d/%d (%.1f%%) | batch %.1fs | "
                "elapsed %.0fs | ETA %.0fs | UNKs this batch %d",
                rg_idx + 1, total_row_groups,
                rows_done, total_rows, pct,
                batch_s, elapsed, eta, unk_count,
            )

    finally:
        if writer is not None:
            writer.close()

    total_elapsed = time.perf_counter() - run_start
    log.info(
        "Done — %d rows translated in %.0fs (%.0f rows/s) | total UNKs: %d (%.4f%%)",
        rows_done,
        total_elapsed,
        rows_done / total_elapsed,
        total_unks,
        100 * total_unks / max(rows_done, 1),
    )


if __name__ == "__main__":
    main()
