"""Two decisive checks before the next big change (~15-20 min):
 A) OWNER CHECK - for our false matches: does the record truly belong to ANOTHER S1 entity (a near-duplicate
    "sibling" of ours) or to nobody (a pure decoy)?  -> decides whether we need S1-vs-S1 competition features.
 B) REVERSE BLOCKING - for each S2/S3 record, query the S1 index: is its true S1 in the record's top-k?
    Recall of forward top-50  UNION  reverse top-k, and how many extra candidates that costs.
Usage: python src/deep_diag.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from blocking import KeyIndex, KEY_COLS, CAPS
from pipeline import CAP_MULT, PQ, OUT, load, say
from train_full import decide

if __name__ == "__main__":
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    ex = gt.assign(cand_id=gt.matched_entity_ids.str.split(",")).explode("cand_id")
    ex = ex[ex.cand_id.notna() & (ex.cand_id != "")]
    owner = dict(zip(ex.cand_id.values, ex.source1_entity_id.values))
    s1raw = pd.read_parquet(f"{PQ}/train_source1.parquet").set_index("entity_id")
    oraw = pd.concat([pd.read_parquet(f"{PQ}/train_source{k}.parquet") for k in "23"]).set_index("entity_id")
    say("ground truth + raw records loaded")

    # ---------------- A) owner check ----------------
    meta = json.load(open(os.path.join(OUT, "lgb_full.json")))
    va = pd.read_parquet(f"{PQ}/full_val_feats.parquet")
    p = lgb.Booster(model_file=os.path.join(OUT, "lgb_full.txt")).predict(va[meta["features"]])
    pred = decide(va, p, meta["threshold"])
    ids = set(pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id)
    truth = {s: set(m.split(",")) if m else set() for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) if s in ids}
    fps = [(s, c) for s, cs in pred.items() for c in cs - truth.get(s, set())]
    kinds = {"belongs to ANOTHER S1": 0, "belongs to NO S1 (decoy)": 0}
    sib_name = []
    rows = []
    for s, c in fps:
        o = owner.get(c)
        if o is None:
            kinds["belongs to NO S1 (decoy)"] += 1
        else:
            kinds["belongs to ANOTHER S1"] += 1
            sib_name.append(fuzz.token_set_ratio(s1raw.loc[s, "business_name"], s1raw.loc[o, "business_name"]))
        rows.append((s, c, o))
    print(f"\nA) OWNER CHECK on {len(fps):,} false matches:")
    for k, v in kinds.items():
        print(f"   {k:28s}: {v:6,} ({v/max(len(fps),1)*100:.1f}%)")
    if sib_name:
        sn = np.array(sib_name)
        print(f"   when another S1 owns it: name similarity (our S1 vs true owner S1): median {np.median(sn):.0f}, "
              f">=90 in {np.mean(sn>=90)*100:.0f}%  (high = near-duplicate siblings in S1)")
    print("\n   12 examples:")
    for s, c, o in rows[:12]:
        print(f"   OUR S1  : {s1raw.loc[s,'business_name']} | {s1raw.loc[s,'business_address']}")
        print(f"   record  : {oraw.loc[c,'business_name']} | {oraw.loc[c,'business_address']}")
        print(f"   TRUE S1 : " + (f"{s1raw.loc[o,'business_name']} | {s1raw.loc[o,'business_address']}" if o else "(none - decoy)") + "\n")

    # ---------------- B) reverse blocking ----------------
    del s1raw, oraw, va, owner; import gc; gc.collect()
    s1k = load("train", 1, KEY_COLS)
    idx = KeyIndex(s1k, caps={k: v * CAP_MULT for k, v in CAPS.items()})
    say("S1 index built")
    cands = pd.read_parquet(f"{PQ}/train_cands.parquet", columns=["s1_row", "o_row", "rank"])
    z = np.load(f"{PQ}/train_cand_ids.npz", allow_pickle=True)
    s1_ids, pool_ids = z["s1"], z["pool"]
    s1_row_of = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    pool_row_of = pd.Series(np.arange(len(pool_ids)), index=pool_ids)
    tv = ex[ex.source1_entity_id.isin(ids)]
    t_pairs = set(zip(s1_row_of[tv.source1_entity_id].to_numpy().tolist(), pool_row_of[tv.cand_id].to_numpy().tolist()))
    samp = np.random.RandomState(0).choice(len(pool_ids), min(200_000, len(pool_ids)), replace=False)
    # keep only the forward pairs we need (val entities' rows, sampled records) -> small memory
    val_rows = np.unique(s1_row_of[list(ids)].to_numpy())
    keep = np.isin(cands.s1_row.to_numpy(), val_rows) | np.isin(cands.o_row.to_numpy(), samp)
    cands = cands[keep]
    fwd = set(zip(cands.s1_row.to_numpy().tolist(), cands.o_row.to_numpy().tolist()))
    say(f"forward candidates loaded: {len(fwd):,} relevant pairs")
    # query: the true-match records of val entities + a random 200k pool sample (to measure the extra cost)
    other = pd.concat([load("train", k, KEY_COLS) for k in "23"], ignore_index=True)
    q_rows = np.unique(np.r_[pool_row_of[tv.cand_id].to_numpy(), samp])
    q = other.iloc[q_rows].reset_index(drop=True)
    r = idx.query(q, k_max=10, chunk=5000, verbose=False, as_rows=True)   # s1_row = row in q, o_row = S1 row
    r["pool_row"] = q_rows[r.s1_row.to_numpy()]
    say("reverse query done")
    print(f"\nB) REVERSE BLOCKING on {len(t_pairs):,} true pairs of the val entities:")
    fwd_hit = np.mean([pr in fwd for pr in t_pairs])
    print(f"   forward top-50 only           : {fwd_hit*100:6.2f}%")
    in_samp = set(samp.tolist())
    for k in [1, 3, 5, 10]:
        rev = set(zip(r.o_row[r["rank"] <= k].to_numpy().tolist(), r.pool_row[r["rank"] <= k].to_numpy().tolist()))
        union = np.mean([(pr in fwd) or (pr in rev) for pr in t_pairs])
        rs = r[(r["rank"] <= k) & r.pool_row.isin(in_samp)]
        new = np.mean([(a, b) not in fwd for a, b in zip(rs.o_row.to_numpy().tolist(), rs.pool_row.to_numpy().tolist())])
        extra_per_s1 = len(rs) / len(samp) * len(pool_ids) * new / len(s1_ids)
        print(f"   forward-50  UNION reverse top-{k:<2}: {union*100:6.2f}%   (+{extra_per_s1:4.1f} new candidates per S1)")
