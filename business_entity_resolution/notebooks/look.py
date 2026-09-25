import os, time
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
print("data folder found:", os.path.isdir(D), flush=True)

files = {
    "train_s1": f"{D}/train/train_source1.tsv",
    "train_s2": f"{D}/train/train_source2.tsv",
    "train_s3": f"{D}/train/train_source3.tsv",
    "train_gt": f"{D}/train/train_ground_truth.tsv",
    "test_s1":  f"{D}/test/test_source1.tsv",
    "test_s2":  f"{D}/test/test_source2.tsv",
    "test_s3":  f"{D}/test/test_source3.tsv",
}

def read(path, n=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=n)

print("\n--- ROW COUNTS ---", flush=True)
for name, path in files.items():
    t = time.time()
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    print(f"{name}: {n}  ({time.time()-t:.1f}s)", flush=True)

print("\n--- SAMPLE ROWS ---", flush=True)
for name in ["train_s1", "train_s2", "train_s3", "train_gt"]:
    print(f"\n=== {name} ===")
    print(read(files[name], 5).to_string(), flush=True)

print("\n--- COUNTRY COUNTS ---", flush=True)
for name in ["train_s1", "test_s1"]:
    print(f"\n{name}:")
    print(read(files[name])["country"].value_counts(), flush=True)