"""Normalise the dev (or train/test) parquet files. Usage: python src/run_normalize.py dev"""
import os, sys, time
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_df

PQ = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")


WORKERS = 1   # 1 = no multiprocessing (safe on Windows). Try 2-3 for the full data if RAM allows.


def run(prefix, workers=WORKERS):
    for k in ["1", "2", "3"]:
        src, out = f"{PQ}/{prefix}_source{k}.parquet", f"{PQ}/{prefix}_source{k}_norm.parquet"
        t = time.time()
        df = pd.read_parquet(src)
        step = 50_000
        chunks = [df.iloc[i:i + step] for i in range(0, len(df), step)]
        if workers > 1:
            with Pool(workers) as p:
                parts = p.map(normalize_df, chunks)
        else:
            parts = [normalize_df(c) for c in chunks]
        df = pd.concat(parts)
        del parts, chunks
        df.to_parquet(out, index=False)
        print(f"{prefix}_source{k}: {len(df):,} rows normalised ({time.time()-t:.0f}s)", flush=True)
    return df


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "dev"
    run(prefix)
    # show 8 random examples per source so we can eyeball the result
    cols = ["business_name", "name_core", "legal", "business_address", "addr_norm", "addr_nums", "postcode", "state"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    for k in ["1", "2", "3"]:
        df = pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet")
        print(f"\n=== {prefix}_source{k} ===")
        print(df[cols].sample(8, random_state=0).to_string(index=False))
        nonascii = df[~df.business_name.map(str.isascii)]
        print(f"\nnon-Latin names: {len(nonascii):,} ({len(nonascii)/len(df)*100:.1f}%)  examples:")
        print(nonascii[["business_name", "name_core", "name_cons"]].head(5).to_string(index=False))
        print(f"empty name_core: {(df.name_core=='').mean()*100:.2f}%   empty addr: {(df.addr_norm=='').mean()*100:.2f}%   "
              f"has postcode: {(df.postcode!='').mean()*100:.1f}%   has state: {(df.state!='').mean()*100:.1f}%")
