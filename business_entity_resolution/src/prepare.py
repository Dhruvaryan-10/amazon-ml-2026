"""Run once: TSV -> parquet, fixed val split, 5% dev sample."""
import os, time
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "dataset")
PQ = os.path.join(ROOT, "data", "parquet")
os.makedirs(PQ, exist_ok=True)
SEED, VAL_FRAC, DEV_FRAC = 42, 0.20, 0.05

# 1. TSV -> parquet (skips files already converted)
for split in ["train", "test"]:
    for f in sorted(os.listdir(f"{RAW}/{split}")):
        out = f"{PQ}/{f.replace('.tsv', '.parquet')}"
        if not f.endswith(".tsv") or os.path.exists(out):
            continue
        t = time.time()
        df = pd.read_csv(f"{RAW}/{split}/{f}", sep="\t", dtype=str, keep_default_na=False)
        df.to_parquet(out, index=False)
        print(f"{f}: {len(df):,} rows -> parquet ({time.time()-t:.0f}s)", flush=True)
        del df

# 2. Fixed split of train S1 entities
rng = np.random.RandomState(SEED)
s1 = pd.read_parquet(f"{PQ}/train_source1.parquet")
gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
s1["is_val"] = rng.rand(len(s1)) < VAL_FRAC
s1["in_dev"] = rng.rand(len(s1)) < DEV_FRAC
s1[["entity_id", "country", "is_val", "in_dev"]].to_parquet(f"{PQ}/split.parquet", index=False)
print(f"\nval entities: {s1.is_val.sum():,}   dev entities: {s1.in_dev.sum():,}")

# 3. Dev sample: dev S1 + ALL their true matches + DEV_FRAC of every other record
dev_s1 = s1[s1.in_dev].drop(columns="in_dev")
dev_gt = gt[gt.source1_entity_id.isin(dev_s1.entity_id)]
dev_matched = {i for x in dev_gt.matched_entity_ids if x for i in x.split(",")}
dev_s1.to_parquet(f"{PQ}/dev_source1.parquet", index=False)
dev_gt.to_parquet(f"{PQ}/dev_ground_truth.parquet", index=False)

for k in ["2", "3"]:
    s = pd.read_parquet(f"{PQ}/train_source{k}.parquet")
    is_match = s.entity_id.isin(dev_matched)
    keep = is_match | (rng.rand(len(s)) < DEV_FRAC)
    s[keep].to_parquet(f"{PQ}/dev_source{k}.parquet", index=False)
    print(f"dev_source{k}: {keep.sum():,} records "
          f"({is_match.sum():,} true matches, {(keep & ~is_match).sum():,} others)")
    del s

print("\nDONE")