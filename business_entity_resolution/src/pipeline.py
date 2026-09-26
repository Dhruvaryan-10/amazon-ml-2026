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
N_TRAIN, N_VAL = int(os.environ.get("N_TRAIN", 80_000)), int(os.environ.get("N_VAL", 25_000))
FEAT_CHUNK = 60_000   # S1 entities per feature batch
SEED = 42
T0 = time.time()


def say(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def load(prefix, k, cols):
    return pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet", columns=cols)


def block(prefix, reuse=False):
    """-> s1 ids, candidate table (int rows) with candidate-side stats, pool ids.
    Cached to data/parquet/{prefix}_cands.parquet; reuse=True skips blocking if the cache exists."""
    cpath, ipath = f"{PQ}/{prefix}_cands.parquet", f"{PQ}/{prefix}_cand_ids.npz"
    if reuse and os.path.exists(cpath):
        c = pd.read_parquet(cpath)
        z = np.load(ipath, allow_pickle=True)
        say(f"reusing cached {prefix} candidates: {len(c):,} pairs")
        return z["s1"], c, z["pool"]
    s1_ids, c, pool_ids = _block(prefix)
    c.to_parquet(cpath, index=False)
    np.savez(ipath, s1=s1_ids, pool=pool_ids)
    return s1_ids, c, pool_ids


COMP_TOP = 3         # competitor S1s per record kept for S1-vs-S1 features (by blocking rank)
REV_K = 5            # reverse blocking: each S2/S3 record proposes its top-5 S1 entities


def _block(prefix):
    """Forward (S1 -> top-K records) UNION reverse (record -> top-REV_K S1) candidates,
    plus candidate-side competition stats over ALL S1."""
    caps = {k: v * CAP_MULT for k, v in CAPS.items()}
    other = pd.concat([load(prefix, k, KEY_COLS) for k in "23"], ignore_index=True)
    say(f"pool loaded: {len(other):,} S2/S3 records")
    idx = KeyIndex(other, caps=caps)
    pool_ids = other.entity_id.to_numpy()
    del other; gc.collect()
    s1 = load(prefix, 1, KEY_COLS)
    say(f"S1 loaded: {len(s1):,}; forward query...")
    c = idx.query(s1, k_max=K, chunk=5000, as_rows=True)
    del idx; gc.collect()
    say(f"forward candidates: {len(c):,} pairs ({len(c)/len(s1):.1f} per S1)")

    # ---- reverse: index S1, query every S2/S3 record ----
    s1_ids = s1.entity_id.to_numpy()
    sidx = KeyIndex(s1, caps=caps)
    del s1; gc.collect()
    other = pd.concat([load(prefix, k, KEY_COLS) for k in "23"], ignore_index=True)
    rev = []
    step = 500_000
    for a in range(0, len(other), step):
        r = sidx.query(other.iloc[a:a + step].reset_index(drop=True), k_max=REV_K, chunk=20000,
                       verbose=False, as_rows=True)
        rev.append(pd.DataFrame({"s1_row": r.o_row.to_numpy(), "o_row": (r.s1_row.to_numpy() + a).astype(np.int32),
                                 "rev_score": r.score.to_numpy(), "rev_rank": r["rank"].to_numpy()}))
        say(f"  reverse query {min(a+step, len(other)):,}/{len(other):,} records")
    del sidx, other; gc.collect()
    rev = pd.concat(rev, ignore_index=True)
    c = c.merge(rev, on=["s1_row", "o_row"], how="outer")
    del rev; gc.collect()
    new = c.score.isna().to_numpy()
    c["score"] = c.score.fillna(0).astype(np.float32)
    c["rank"] = c["rank"].fillna(K + 1).astype(np.int16)
    c["rev_score"] = c.rev_score.fillna(0).astype(np.float32)
    c["rev_rank"] = c.rev_rank.fillna(REV_K + 1).astype(np.int16)
    c = c.sort_values(["s1_row", "rank", "rev_rank"], kind="stable").reset_index(drop=True)
    say(f"forward UNION reverse: {len(c):,} pairs ({new.sum():,} added by reverse; {len(c)/len(s1_ids):.1f} per S1)")

    # candidate-side stats over ALL S1: rank of this S1 among the S1s that picked this record,
    # how many S1s compete, and the score gap to the best OTHER S1 (same-name siblings)
    sc = (c.score.to_numpy() + c.rev_score.to_numpy()).astype(np.float32)
    o = np.lexsort((-sc, c.o_row.to_numpy()))
    orow, scs = c.o_row.to_numpy()[o], sc[o]
    first = np.r_[0, np.flatnonzero(np.diff(orow)) + 1]
    sizes = np.diff(np.r_[first, len(orow)])
    top = np.repeat(scs[first], sizes)
    second = np.repeat(np.where(sizes > 1, scs[np.minimum(first + 1, len(scs) - 1)], 0), sizes)
    cr = np.arange(len(orow)) - np.repeat(first, sizes) + 1
    gap = np.where(cr == 1, scs - second, scs - top)          # >0: this S1 beats every other S1
    ntie = np.repeat(np.add.reduceat((scs >= 0.9 * np.repeat(scs[first], sizes)).astype(np.int32), first), sizes)
    for name, arr, dt in [("cand_rank", cr, np.int32), ("cand_n_s1", np.repeat(sizes, sizes), np.int32),
                          ("cand_gap", gap, np.float32), ("cand_n_tie", ntie, np.int32)]:
        out = np.empty(len(c), dtype=dt); out[o] = arr; c[name] = out
    return s1_ids, c, pool_ids


FCOLS = ["entity_id", "business_name"] + NAME_COLS + ADDR_COLS


def load_tables(prefix):
    """Normalised columns needed for features, kept as compact Arrow tables (row order = pool order),
    plus name-token document frequencies over the whole S2/S3 pool."""
    s1t = pq.read_table(f"{PQ}/{prefix}_source1_norm.parquet", columns=FCOLS)
    ot = pa.concat_tables([pq.read_table(f"{PQ}/{prefix}_source{k}_norm.parquet", columns=FCOLS) for k in "23"])
    toks = pc.list_flatten(pc.utf8_split_whitespace(ot["name_core"].combine_chunks()))
    vc = pc.value_counts(toks)
    tokfreq = dict(zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()))
    return s1t, ot, tokfreq


def s1_stats(cands, n_s1):
    """Per-S1 max blocking score and #candidates over its FULL candidate list (arrays indexed by s1_row)."""
    smax = np.zeros(n_s1, dtype=np.float32)
    np.maximum.at(smax, cands.s1_row.to_numpy(), cands.score.to_numpy())
    return smax, np.bincount(cands.s1_row.to_numpy(), minlength=n_s1).astype(np.int32)


def features_for_pairs(tables, s1_ids, sub, pool_ids, stats, return_recs=False):
    """Features for an explicit subset of candidate pairs (rows of the candidate table)."""
    s1t, ot, tokfreq = tables
    sub = sub.reset_index(drop=True).copy()
    sub["s1_id"] = s1_ids[sub.s1_row.to_numpy()]
    sub["cand_id"] = pool_ids[sub.o_row.to_numpy()]
    sub["s1_max"] = stats[0][sub.s1_row.to_numpy()]
    sub["s1_n"] = stats[1][sub.s1_row.to_numpy()]
    s1n = s1t.take(pa.array(np.unique(sub.s1_row.to_numpy()))).to_pandas()
    on = ot.take(pa.array(np.unique(sub.o_row.to_numpy()))).to_pandas()
    X = pair_features(sub, s1n, on, tokfreq=tokfreq, n_pool=len(ot))
    if return_recs:
        return sub[["s1_id", "cand_id"]], X, on.drop_duplicates("entity_id").set_index("entity_id")
    return sub[["s1_id", "cand_id"]], X


def features_for(tables, s1_ids, cands, pool_ids, s1_rows, return_recs=False, stats=None):
    """Features for all candidate pairs of the given S1 rows."""
    sub = cands[np.isin(cands.s1_row.to_numpy(), s1_rows)]
    if stats is None:
        stats = s1_stats(cands, len(s1_ids))
    return features_for_pairs(tables, s1_ids, sub, pool_ids, stats, return_recs)


def label(pairs, truth_pairs):
    return np.array([(s, c) in truth_pairs for s, c in zip(pairs.s1_id.values, pairs.cand_id.values)], dtype=np.int8)


def run_train(reuse=False):
    s1_ids, cands, pool_ids = block("train", reuse)
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
    stats = s1_stats(cands, len(s1_ids))
    say("feature tables loaded")
    # COMPETITORS: for the plausible records of the sampled S1s, the top other S1s claiming the same record.
    # Their stage-1 probabilities let stage-2 compare S1 against S1 (same-name siblings, 1:1 ownership).
    sampled = np.zeros(len(s1_ids), dtype=bool); sampled[tr_rows] = True; sampled[va_rows] = True
    cs1, cor = cands.s1_row.to_numpy(), cands.o_row.to_numpy()
    plaus = sampled[cs1] & ((cands["rank"].to_numpy() <= 10) | (cands.rev_rank.to_numpy() <= 5))
    rec_mask = np.zeros(len(pool_ids), dtype=bool); rec_mask[cor[plaus]] = True
    comp = cands[rec_mask[cor] & ~sampled[cs1] & (cands.cand_rank.to_numpy() <= COMP_TOP)]
    say(f"competitor pairs: {len(comp):,}")
    comp_ids = set(s1_ids[np.unique(comp.s1_row.to_numpy())])
    gtc = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gtc = gtc[gtc.source1_entity_id.isin(comp_ids)]
    truth_pairs |= {(s, c) for s, m in zip(gtc.source1_entity_id, gtc.matched_entity_ids) if m for c in m.split(",")}
    for name, r in [("full_train", tr_rows), ("full_val", va_rows)]:
        own = cands[np.isin(cs1, r)]
        recs = np.zeros(len(pool_ids), dtype=bool); recs[own.o_row.to_numpy()] = True
        cp = comp[recs[comp.o_row.to_numpy()]]
        parts = []
        for flag, sub in [(0, own), (1, cp)]:
            for a in range(0, len(sub), 2_000_000):
                pairs, X = features_for_pairs(tables, s1_ids, sub.iloc[a:a + 2_000_000], pool_ids, stats)
                X["label"] = label(pairs, truth_pairs)
                X["is_comp"] = np.int8(flag)
                parts.append(pd.concat([pairs, X], axis=1))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(f"{PQ}/{name}_feats.parquet", index=False)
        own_n = int((df.is_comp == 0).sum())
        say(f"{name}: {own_n:,} own pairs (+{len(df)-own_n:,} competitor pairs), "
            f"positive rate {df.label[df.is_comp == 0].mean()*100:.1f}% -> saved")
        del df, parts; gc.collect()
    pd.DataFrame({"entity_id": s1_ids[va_rows]}).to_parquet(f"{PQ}/full_val_ids.parquet", index=False)
    from explog import log
    log(f"full-density blocking CAP_MULT={CAP_MULT} K={K}", notes="train pool, 25k val S1",
        blocking_recall=len(tv & got) / len(tv))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        run_train(reuse="reuse" in sys.argv)
    else:
        from predict import run_test
        run_test(reuse="reuse" in sys.argv)
