"""Normalise source files, streaming in chunks so memory stays low.

Usage:  python src/run_normalize.py dev      (dev sample, prints examples)
        python src/run_normalize.py train    (full train: ~10 min)
        python src/run_normalize.py test     (full test:  ~10 min)
"""
import os, sys, time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_df

PQ = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")
STEP = 100_000


def run(prefix):
    for k in ["1", "2", "3"]:
        src, out = f"{PQ}/{prefix}_source{k}.parquet", f"{PQ}/{prefix}_source{k}_norm.parquet"
        t = time.time()
        pf = pq.ParquetFile(src)
        writer, n = None, 0
        for batch in pf.iter_batches(batch_size=STEP):
            df = normalize_df(batch.to_pandas())
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table.cast(writer.schema))
            n += len(df)
            if n % 1_000_000 < STEP:
                print(f"   {prefix}_source{k}: {n:,} rows ({time.time()-t:.0f}s)", flush=True)
        writer.close()
        print(f"{prefix}_source{k}: {n:,} rows normalised ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "dev"
    run(prefix)
    if prefix != "dev":
        sys.exit()
    cols = ["business_name", "name_core", "legal", "business_address", "addr_norm", "addr_nums", "postcode", "state"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    for k in ["1", "2", "3"]:
        df = pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet")
        print(f"\n=== {prefix}_source{k} ===")
        print(df[cols].sample(8, random_state=0).to_string(index=False))
        nonascii = df[~df.business_name.map(str.isascii)]
        print(f"\nnon-Latin names: {len(nonascii):,} ({len(nonascii)/len(df)*100:.1f}%)  examples:")
        print(nonascii[["business_name", "name_core", "name_cons"]].head(5).to_string(index=False))
