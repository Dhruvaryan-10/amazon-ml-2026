"""WHY does blocking miss true matches at full density?  (~10 min)
Queries the 25k val entities against the full train pool with K=300 and classifies every true pair:
  - ranked 1-50        -> found (what we have now)
  - ranked 51-300      -> "rank miss": shares keys but look-alikes outrank it  -> fix: better ranking / per-source K
  - not in top 300     -> "key miss": shares no usable key                     -> fix: new keys / higher caps
Usage: python src/diagnose.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from blocking import KeyIndex, KEY_COLS, CAPS, _keys_of
from pipeline import CAP_MULT, PQ, load, say

if __name__ == "__main__":
    other = pd.concat([load("train", k, KEY_COLS) for k in "23"], ignore_index=True)
    caps = {k: v * CAP_MULT for k, v in CAPS.items()}
    idx = KeyIndex(other, caps=caps)
    val_ids = set(pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id)
    s1 = load("train", 1, KEY_COLS)
    s1 = s1[s1.entity_id.isin(val_ids)].reset_index(drop=True)
    say(f"querying {len(s1):,} val entities with K=300")
    c = idx.query(s1, k_max=300, chunk=2000, verbose=False)
    c["src"] = c.cand_id.str[:2]
    c["rank_src"] = c.groupby(["s1_id", "src"]).cumcount() + 1

    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(val_ids)]
    t = gt.assign(cand_id=gt.matched_entity_ids.str.split(",")).explode("cand_id")
    t = t[t.cand_id.notna() & (t.cand_id != "")][["source1_entity_id", "cand_id"]].rename(columns={"source1_entity_id": "s1_id"})
    m = t.merge(c[["s1_id", "cand_id", "rank", "rank_src"]], on=["s1_id", "cand_id"], how="left")
    r = m["rank"]
    from explog import log
    log("blocking v4 diagnose (full density, 25k val)", notes=f"rank51-300={((r>50)&(r<=300)).mean():.4f}; keymiss={r.isna().mean():.4f}",
        blocking_recall=(r <= 50).mean(), avg_candidates=round((c['rank'] <= 50).groupby(c.s1_id).sum().mean(), 1))
    print(f"\ntrue pairs: {len(m):,}")
    print(f"  rank 1-50     : {(r <= 50).mean()*100:6.2f}%   <- current recall")
    print(f"  rank 51-100   : {((r > 50) & (r <= 100)).mean()*100:6.2f}%")
    print(f"  rank 101-300  : {((r > 100) & (r <= 300)).mean()*100:6.2f}%   <- rank misses")
    print(f"  not in top 300: {r.isna().mean()*100:6.2f}%   <- key misses")

    print("\nper-source top-K (take the best K from S2 AND the best K from S3):")
    for k in [25, 30, 40, 50]:
        rec = (m.rank_src <= k).mean() * 100
        avg = (c.rank_src <= k).groupby(c.s1_id).sum().mean()
        print(f"  top {k:>2} per source: recall {rec:6.2f}%   avg candidates {avg:5.1f}")

    # key misses: do they share ANY key if we ignore the caps? which types?
    miss = m[r.isna()]
    print(f"\nkey misses: {len(miss):,}. Shared key types when caps are ignored (sample of 2000):")
    s1i = s1.set_index("entity_id")
    oi = other.set_index("entity_id")
    kc = [k for k in KEY_COLS[1:]]
    typ, none = {}, 0
    rows = []
    for s, o in miss.sample(min(2000, len(miss)), random_state=0)[["s1_id", "cand_id"]].values:
        a = _keys_of(*s1i.loc[s, kc]); b = _keys_of(*oi.loc[o, kc])
        shared = a & b
        if not shared:
            none += 1
        for k in shared:
            j = np.searchsorted(idx.uniq, hash(k))
            df = int(idx.count[j]) if j < len(idx.uniq) and idx.uniq[j] == hash(k) else 0
            typ.setdefault(k[0], []).append(df)
        rows.append((s, o, len(shared)))
    print(f"  share NO key at all: {none/max(len(rows),1)*100:.1f}%")
    for k, v in sorted(typ.items(), key=lambda x: -len(x[1])):
        v = np.array(v)
        print(f"  type {k}: in {len(v)/max(len(rows),1)*100:5.1f}% of misses; pool frequency median {int(np.median(v))}, "
              f"cap {caps[k]} -> {np.mean(v > caps[k])*100:.0f}% over cap")

    s1f = pd.read_parquet(f"{PQ}/train_source1.parquet").set_index("entity_id")
    of = pd.concat([pd.read_parquet(f"{PQ}/train_source{k}.parquet") for k in "23"]).set_index("entity_id")
    print("\n12 key misses:")
    for s, o, n in rows[:12]:
        print(f"  S1 {s1f.loc[s, 'business_name']} | {s1f.loc[s, 'business_address']}")
        print(f"  -> {of.loc[o, 'business_name']} | {of.loc[o, 'business_address']}   (shared keys: {n})\n")
    print("12 rank misses (rank 51-300):")
    rm = m[(r > 50)].sample(min(12, int((r > 50).sum())), random_state=0)
    for s, o, rk in rm[["s1_id", "cand_id", "rank"]].values:
        print(f"  S1 {s1f.loc[s, 'business_name']} | {s1f.loc[s, 'business_address']}")
        print(f"  -> {of.loc[o, 'business_name']} | {of.loc[o, 'business_address']}   (rank {int(rk)})\n")
