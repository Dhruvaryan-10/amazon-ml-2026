"""Score every test candidate pair and write the two submission files.
  python src/pipeline.py test          -> block + score (blocking is cached to data/parquet/test_cands.parquet)
  python src/pipeline.py test reuse    -> skip blocking, reuse the cached candidates (saves ~35 min)
Uses the stage-2 model (lgb_stack.txt) when train_stack.py found it better, else stage-1 only.
"""
import os, sys, json, gc, subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
from stack import stack_features


def run_test(reuse=False):
    meta1 = json.load(open(os.path.join(P.OUT, "lgb_full.json")))
    m1 = lgb.Booster(model_file=os.path.join(P.OUT, "lgb_full.txt"))
    sj = os.path.join(P.OUT, "lgb_stack.json")
    meta2 = json.load(open(sj)) if os.path.exists(sj) else {"use_stack": False}
    m2 = lgb.Booster(model_file=os.path.join(P.OUT, "lgb_stack.txt")) if meta2["use_stack"] else None
    thr = meta2["threshold"] if m2 else meta1["threshold"]
    P.say(f"model: {'stage-1 + stage-2' if m2 else 'stage-1 only'}, threshold {thr}")

    s1_ids, cands, pool_ids = P.block("test", reuse)
    tables = P.load_tables("test")
    P.say("feature tables loaded; scoring in chunks")
    probs = np.empty(len(cands), dtype=np.float32)
    cs1 = cands.s1_row.to_numpy()
    for start in range(0, len(s1_ids), P.FEAT_CHUNK):
        rows = np.arange(start, min(start + P.FEAT_CHUNK, len(s1_ids)))
        mask = (cs1 >= rows[0]) & (cs1 <= rows[-1])
        pairs, X, recs = P.features_for(tables, s1_ids, cands, pool_ids, rows, return_recs=True)
        p = m1.predict(X[meta1["features"]])
        if m2 is not None:
            S = stack_features(pairs, p, recs)
            X2 = pd.concat([X[meta2["f1"]].reset_index(drop=True), S.reset_index(drop=True)], axis=1)
            p = m2.predict(X2[meta2["f2"]])
        probs[np.flatnonzero(mask)] = p
        P.say(f"  scored S1 {rows[-1]+1:,}/{len(s1_ids):,}")
    cands["p"] = probs
    del tables; gc.collect()

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
