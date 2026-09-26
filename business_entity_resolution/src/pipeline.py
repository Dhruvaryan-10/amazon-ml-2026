"""Full-scale pipeline: blocking over the WHOLE S2/S3 pool, candidate-side stats over ALL S1,
then features (and, for test, predictions) in chunks so memory stays bounded.

Usage:
  python src/pipeline.py train   -> features for 80k train + 25k val S1 entities, realistic density
  python src/pipeline.py test    -> scores every test pair with outputs/lgb_full.txt, writes submission
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from blocking import KeyIndex, KEY_COLS, CAPS
from features import pair_features, NAME_COLS, ADDR_COLS

ROOT = os.path.join(os.path.dirname(__file__), "..")
PQ = os.path.join(ROOT, "data", "parquet")
OUT = os.path.join(ROOT, "outputs")
CAP_MULT = 3          # key frequency caps are tuned on the 5% dev sample; the full pool is ~12x denser
K = 50
N_TRAIN, N_VAL = 80_000, 25_000
FEAT_CHUNK = 60_000   # S1 entities per feature batch
SEED = 42
T0 = time.time()


def say(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def load(prefix, k, cols):
    return pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet", columns=cols)


def block(prefix):
    """-> s1 key frame, candidate table (int rows) with candidate-side stats, pool ids."""
    other = pd.concat([load(prefix, k, KEY_COLS) for k in "23"], ignore_index=True)
    say(f"pool loaded: {len(other):,} S2/S3 records")
    idx = KeyIndex(other, caps={k: v * CAP_MULT for k, v in CAPS.items()})
    pool_ids = other.entity_id.to_numpy()
    del other; gc.collect()
    s1 = load(prefix, 1, KEY_COLS)
    say(f"S1 loaded: {len(s1):,}; querying...")
    c = idx.query(s1, k_max=K, chunk=5000, as_rows=True)
    del idx; gc.collect()
    say(f"candidates: {len(c):,} pairs ({len(c)/len(s1):.1f} per S1)")
    # candidate-side stats over ALL S1: rank of this S1 among S1s that picked this record
    o = np.lexsort((-c.score.to_numpy(), c.o_row.to_numpy()))
    orow = c.o_row.to_numpy()[o]
    first = np.r_[0, np.flatnonzero(np.diff(orow)) + 1]
    sizes = np.diff(np.r_[first, len(orow)])
    cr = np.empty(len(c), dtype=np.int32); cn = np.empty(len(c), dtype=np.int32)
    cr[o] = np.arange(len(orow)) - np.repeat(first, sizes) + 1
    cn[o] = np.repeat(sizes, sizes)
    c["cand_rank"], c["cand_n_s1"] = cr, cn
    return s1.entity_id.to_numpy(), c, pool_ids


FCOLS = ["entity_id", "business_name"] + NAME_COLS + ADDR_COLS


def load_tables(prefix):
    """Normalised columns needed for features, kept as compact Arrow tables (row order = pool order)."""
    s1t = pq.read_table(f"{PQ}/{prefix}_source1_norm.parquet", columns=FCOLS)
    ot = pa.concat_tables([pq.read_table(f"{PQ}/{prefix}_source{k}_norm.parquet", columns=FCOLS) for k in "23"])
    return s1t, ot


def features_for(tables, s1_ids, cands, pool_ids, s1_rows):
    """Features for the candidate pairs of the given S1 rows."""
    s1t, ot = tables
    sub = cands[np.isin(cands.s1_row.to_numpy(), s1_rows)].reset_index(drop=True)
    sub["s1_id"] = s1_ids[sub.s1_row.to_numpy()]
    sub["cand_id"] = pool_ids[sub.o_row.to_numpy()]
    s1n = s1t.take(pa.array(np.unique(sub.s1_row.to_numpy()))).to_pandas()
    on = ot.take(pa.array(np.unique(sub.o_row.to_numpy()))).to_pandas()
    X = pair_features(sub, s1n, on)
    return sub[["s1_id", "cand_id"]], X


def label(pairs, truth_pairs):
    return np.array([(s, c) in truth_pairs for s, c in zip(pairs.s1_id.values, pairs.cand_id.values)], dtype=np.int8)


def run_train():
    s1_ids, cands, pool_ids = block("train")
    split = pd.read_parquet(f"{PQ}/split.parquet").set_index("entity_id").loc[s1_ids]
    rng = np.random.RandomState(SEED)
    rows = np.arange(len(s1_ids))
    tr_pool, va_pool = rows[~split.is_val.to_numpy()], rows[split.is_val.to_numpy()]
    tr_rows = np.sort(rng.choice(tr_pool, min(N_TRAIN, len(tr_pool)), replace=False))
    va_rows = np.sort(rng.choice(va_pool, min(N_VAL, len(va_pool)), replace=False))
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(s1_ids[np.r_[tr_rows, va_rows]]))]
    truth_pairs = {(s, c) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) if m for c in m.split(",")}
    # recall at full density, on the val entities
    va_ids = set(s1_ids[va_rows])
    tv = {(s, c) for (s, c) in truth_pairs if s in va_ids}
    got = cands[np.isin(cands.s1_row.to_numpy(), va_rows)]
    got = set(zip(s1_ids[got.s1_row.to_numpy()], pool_ids[got.o_row.to_numpy()]))
    say(f"FULL-DENSITY blocking recall on val entities: {len(tv & got)/len(tv)*100:.2f}%")
    tables = load_tables("train")
    say("feature tables loaded")
    for name, r in [("full_train", tr_rows), ("full_val", va_rows)]:
        pairs, X = features_for(tables, s1_ids, cands, pool_ids, r)
        X["label"] = label(pairs, truth_pairs)
        pd.concat([pairs, X], axis=1).to_parquet(f"{PQ}/{name}_feats.parquet", index=False)
        say(f"{name}: {len(X):,} pairs, positive rate {X.label.mean()*100:.1f}% -> saved")
    pd.DataFrame({"entity_id": s1_ids[va_rows]}).to_parquet(f"{PQ}/full_val_ids.parquet", index=False)
    from explog import log
    log(f"full-density blocking CAP_MULT={CAP_MULT} K={K}", notes="train pool, 25k val S1",
        blocking_recall=len(tv & got) / len(tv))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        run_train()
    else:
        from predict import run_test
        run_test()
