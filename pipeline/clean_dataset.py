#!/usr/bin/env python3
"""
Remove noise tokens from labeled_dataset.parquet and write labeled_dataset_clean.parquet.

Noise includes:
  - Log header artifacts: tokens containing '[' or ']' (e.g. '[SpiderMonkey Log', 'Return[V8Log]')
  - Single-character tokens: 'S', 'C', 'L', 'I', etc.
  - Truncated tokens from V8 log line truncation

Keep rule: token must match ^[A-Za-z][A-Za-z0-9_]{1,}$ (starts with letter, ≥2 chars, alphanum/_)
"""
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOGS_DIR    = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE  = LOGS_DIR / "labeled_dataset.parquet"
OUTPUT_FILE = LOGS_DIR / "labeled_dataset_clean.parquet"
BATCH_SIZE  = 200_000

VALID_TOKEN = re.compile(r'^[A-Za-z][A-Za-z0-9_]{1,}$')


def clean_bytecodes(bytecodes: str) -> str:
    if not bytecodes:
        return ""
    tokens = bytecodes.split(",")
    clean = [t for t in tokens if VALID_TOKEN.match(t)]
    return ",".join(clean)


def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"[ERROR] Input not found: {INPUT_FILE}")
    if OUTPUT_FILE.exists():
        print(f"[WARN] Output already exists, overwriting: {OUTPUT_FILE}")

    pf = pq.ParquetFile(INPUT_FILE)
    schema = pf.schema_arrow

    total_rows_in  = 0
    total_rows_out = 0
    total_tokens_in  = 0
    total_tokens_out = 0
    noise_tokens = 0

    writer = None
    batch_num = 0

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Schema: {schema}\n")

    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            batch_num += 1
            rows_in = len(batch)
            total_rows_in += rows_in

            tbl = pa.Table.from_batches([batch])

            bytecodes_col = tbl.column("bytecodes").to_pylist()

            cleaned = []
            batch_tok_in = 0
            batch_tok_out = 0

            for raw in bytecodes_col:
                if raw is None:
                    cleaned.append(None)
                    continue
                tokens = [t.strip() for t in raw.split(",")] if raw else []
                valid  = [t for t in tokens if VALID_TOKEN.match(t)]
                batch_tok_in  += len(tokens)
                batch_tok_out += len(valid)
                # Rejoin without spaces — normalises SM format ("A, B" → "A,B")
                cleaned.append(",".join(valid) if valid else None)

            total_tokens_in  += batch_tok_in
            total_tokens_out += batch_tok_out
            noise_tokens     += batch_tok_in - batch_tok_out

            col_idx = tbl.schema.get_field_index("bytecodes")
            arrays  = [
                tbl.column(i) if i != col_idx else pa.array(cleaned, type=pa.string())
                for i in range(tbl.num_columns)
            ]
            tbl_clean = pa.table(
                {tbl.schema.field(i).name: arrays[i] for i in range(tbl.num_columns)},
                schema=tbl.schema,
            )

            # Drop rows where bytecodes became None (all tokens were noise)
            mask = pa.array([v is not None for v in cleaned], type=pa.bool_())
            tbl_clean = tbl_clean.filter(mask)
            rows_out = len(tbl_clean)
            total_rows_out += rows_out

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    schema=tbl_clean.schema,
                    compression="zstd",
                )

            writer.write_table(tbl_clean)

            if batch_num % 10 == 0:
                pct = total_rows_in / pf.metadata.num_rows * 100
                print(
                    f"  {total_rows_in:>12,} rows processed ({pct:.1f}%) "
                    f"| noise tokens so far: {noise_tokens:,}"
                )
    finally:
        if writer:
            writer.close()

    print(f"\n{'─'*60}")
    print(f"Rows  : {total_rows_in:>12,}  →  {total_rows_out:>12,}  "
          f"(dropped {total_rows_in - total_rows_out:,})")
    print(f"Tokens: {total_tokens_in:>12,}  →  {total_tokens_out:>12,}  "
          f"(removed {noise_tokens:,}, {noise_tokens/max(total_tokens_in,1)*100:.4f}%)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
