"""Score every test candidate pair and write the two submission files.
  python src/pipeline.py test          -> block + score (blocking cached to data/parquet/test_cands.parquet)
  python src/pipeline.py test reuse    -> reuse cached candidates (only if blocking code is unchanged!)
Pass 1 (chunks of S1): features -> stage-1 p1 for every pair; for plausible pairs (p1 >= P_MIN) keep the
          stage-2 inputs that need strings (S2<->S3 agreement) plus the blocking columns.
Pass 2 (whole table, numbers only): S1-vs-S1 competition by probability per record -> stage-2 -> decision.
"""
import os, sys, json, gc, subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from stack import stack_features, comp_features, BLK_KEEP


def run_test(reuse=False):
    meta1 = json.load(open(os.path.join(P.OUT, "lgb_full.json")))
    m1 = lgb.Booster(model_file=os.path.join(P.OUT, "lgb_full.txt"))
    sj = os.path.join(P.OUT, "lgb_stack.json")
    meta2 = json.load(open(sj)) if os.path.exists(sj) else {"use_stack": False}
    m2 = lgb.Booster(model_file=os.path.join(P.OUT, "lgb_stack.txt")) if meta2["use_stack"] else None
    thr = meta2["threshold"] if m2 else meta1["threshold"]
    p_min = meta2.get("p_min", 0.01)
    P.say(f"model: {'stage-1 + stage-2' if m2 else 'stage-1 only'}, threshold {thr}")

    s1_ids, cands, pool_ids = P.block("test", reuse)
    stats = P.s1_stats(cands, len(s1_ids))
    tables = P.load_tables("test")
    P.say("feature tables loaded; pass 1 (features + stage-1) in chunks")
    p1 = np.zeros(len(cands), dtype=np.float32)
    kept_idx, kept_X = [], []
    cs1 = cands.s1_row.to_numpy()
    for start in range(0, len(s1_ids), P.FEAT_CHUNK):
        rows = np.arange(start, min(start + P.FEAT_CHUNK, len(s1_ids)))
        pos = np.flatnonzero((cs1 >= rows[0]) & (cs1 <= rows[-1]))
        pairs, X, recs = P.features_for_pairs(tables, s1_ids, cands.iloc[pos], pool_ids, stats, return_recs=True)
        p = m1.predict(X[meta1["features"]]).astype(np.float32)
        p1[pos] = p
        if m2 is not None:
            k = p >= p_min
            S = stack_features(pairs, p, recs)
            blk = X[[c for c in BLK_KEEP if c in X]]
            kept_idx.append(pos[k])
            kept_X.append(pd.concat([S.reset_index(drop=True), blk.reset_index(drop=True)], axis=1)[k]
                          .astype(np.float32).reset_index(drop=True))
        P.say(f"  scored S1 {rows[-1]+1:,}/{len(s1_ids):,}")
    del tables; gc.collect()

    prob = p1.copy()
    if m2 is not None:
        P.say("pass 2: S1-vs-S1 competition + stage-2")
        idx = np.concatenate(kept_idx)
        K = pd.concat(kept_X, ignore_index=True)
        del kept_X; gc.collect()
        comp = comp_features(cands.o_row.to_numpy().astype(np.int64), p1)
        K.insert(0, "p1", p1[idx])
        for c, v in comp.items():
            K[c] = v[idx]
        del comp; gc.collect()
        prob[idx] = m2.predict(K[meta2["f2"]])
        P.say(f"stage-2 re-scored {len(idx):,} plausible pairs")
    cands["p"] = prob

    best = cands.groupby("o_row").p.transform("max").to_numpy()
    m = cands[(cands.p.to_numpy() == best) & (cands.p.to_numpy() >= thr)]
    sub_dir = os.path.join(P.OUT, "submission")
    os.makedirs(sub_dir, exist_ok=True)

    def write(df, col, path):
        lists = df.groupby("s1_row").o_row.agg(lambda r: ",".join(pool_ids[r.to_numpy()]))
        full = pd.Series("", index=np.arange(len(s1_ids)))
        full.loc[lists.index] = lists.values
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for sid, ids in zip(s1_ids, full.values):
                f.write(f"{sid}\t{ids}\n")
    write(m, "matched_entity_ids", os.path.join(sub_dir, "matching_results.tsv"))
    write(cands, "candidate_entity_ids", os.path.join(sub_dir, "candidate_pairs.tsv"))
    n_match = m.groupby("s1_row").size()
    P.say(f"written: {len(m):,} matches; {len(n_match):,} of {len(s1_ids):,} S1 have >=1 match "
          f"({(1-len(n_match)/len(s1_ids))*100:.1f}% predicted singletons); threshold {thr}")
    val = os.path.join(P.ROOT, "utils", "validate_submission.py")
    test_dir = os.path.join(P.ROOT, "data", "dataset", "test")
    subprocess.run([sys.executable, val, "--matching", os.path.join(sub_dir, "matching_results.tsv"),
                    "--candidate", os.path.join(sub_dir, "candidate_pairs.tsv"), "--test-dir", test_dir])
