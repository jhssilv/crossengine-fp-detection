"""
Pre-compute ground-truth per-script statistics from translated_dataset.parquet.

For each script_url, counts:
  - fp_count:  ground-truth FP functions in that script
  - total:     total functions in that script

Saves ir/script_stats.json: {script_url: {"fp_count": int, "total": int}}

Computed from the full 68M-row population before any train/test split, so
safe to use as a feature without leakage (analogous to compute_script_density.py).

Usage:
    python ir/scripts/compute_script_stats.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

_IR_ROOT = Path(__file__).parent.parent
DATASET  = _IR_ROOT / "translated_dataset.parquet"
OUT      = _IR_ROOT / "script_stats.json"


def main():
    pf = pq.ParquetFile(DATASET)
    n_rg       = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    fp_count  = defaultdict(int)
    tot_count = defaultdict(int)

    t0 = time.time()
    rows_done = 0
    print(f"Streaming {total_rows:,} rows across {n_rg} row groups...")

    for rg_idx in range(n_rg):
        if rg_idx % 100 == 0:
            elapsed = time.time() - t0
            pct = 100 * rows_done / total_rows if rows_done else 0
            eta = (elapsed / rows_done) * (total_rows - rows_done) if rows_done else 0
            print(f"  RG {rg_idx}/{n_rg}  {rows_done:,}/{total_rows:,} ({pct:.1f}%)  "
                  f"scripts={len(tot_count):,}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        df = pf.read_row_group(rg_idx, columns=["script_url", "is_fp"]).to_pandas()
        df["is_fp"]      = df["is_fp"].fillna(False).astype(bool)
        df["script_url"] = df["script_url"].fillna("")

        for url, is_fp in zip(df["script_url"].tolist(), df["is_fp"].tolist()):
            tot_count[url] += 1
            if is_fp:
                fp_count[url] += 1

        rows_done += len(df)

    elapsed = time.time() - t0
    print(f"\nDone: {rows_done:,} rows in {elapsed:.0f}s ({rows_done/elapsed:.0f} rows/s)")
    print(f"Unique scripts : {len(tot_count):,}")
    print(f"Scripts with ≥1 FP function: {sum(1 for u in tot_count if fp_count[u] > 0):,}")

    stats = {url: {"fp_count": fp_count[url], "total": tot_count[url]}
             for url in tot_count}

    with open(OUT, "w") as f:
        json.dump(stats, f)
    print(f"Saved → {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
