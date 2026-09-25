"""Candidate generation (blocking) by weighted key overlap.

Every record emits "keys". Two records sharing a rare key are likely the same business:
  n  name token (or its phonetic skeleton) within the same country+state
  N  name token within the country (catches missing/wrong state) - stricter frequency cap
  p  first 5 letters of the space-less name, same country+state ("southerneducational" ~ "southern ...")
  a  house number + address word, same country ("1500|jupiter")
  P  PAIR of name tokens, whole country ("bengaluru+infra") - rare even when single words are
     common, and works when one record has no state or no address
  e  the full space-less name, whole country ("bengaluruinfra")
Keys that are too common are dropped (frequency cap). Each S1 entity then gets its
top-K S2/S3 records ranked by the sum of IDF weights of shared keys.

Usage:  python src/blocking.py dev          (dev sample, prints recall)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
PQ = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")

CAPS = {"n": 300, "N": 40, "p": 150, "a": 150, "P": 150, "e": 150}   # max #S2/S3 records per key (dev scale)
WEIGHT = {"n": 1.0, "N": 0.6, "p": 0.8, "a": 1.2, "P": 1.0, "e": 1.0}
K_MAX = 100          # candidates kept per S1 entity (we measure recall at several K)
CHUNK = 5000         # S1 entities per batch (memory control)


def record_keys(df):
    """Return a DataFrame (row, key) with every key each record emits."""
    rows, keys = [], []
    for i, (c, st, core, cons, nospace, nums, words) in enumerate(zip(
            df.country.values, df.state.values, df.name_core.values, df.name_cons.values,
            df.name_nospace.values, df.addr_nums.values, df.addr_words.values)):
        ks = set()
        for t in set(core.split()) | set(cons.split()):
            if len(t) >= 3 or (t.isdigit() and len(t) >= 2):
                ks.add(f"n|{c}|{st}|{t}")
                ks.add(f"N|{c}|{t}")
        if len(nospace) >= 5:
            ks.add(f"p|{c}|{st}|{nospace[:5]}")
        if len(nospace) >= 4:
            ks.add(f"e|{c}|{nospace}")
        for toks in (core.split(), cons.split()):
            toks = sorted({t for t in toks if len(t) >= 2})[:8]
            for x in range(len(toks)):
                for y in range(x + 1, len(toks)):
                    ks.add(f"P|{c}|{toks[x]}|{toks[y]}")
        ws = [w for w in words.split() if len(w) >= 4]
        for n in nums.split():
            if len(n) >= 2 or n != "0":
                for w in ws:
                    ks.add(f"a|{c}|{n}|{w}")
        rows.extend([i] * len(ks))
        keys.extend(ks)
    return pd.DataFrame({"row": np.array(rows, dtype=np.int32), "key": keys})


def generate(s1, other, caps=CAPS, k_max=K_MAX, verbose=True):
    """s1, other: normalised dataframes. Returns DataFrame(s1_id, cand_id, score, rank)."""
    t0 = time.time()
    ko = record_keys(other)
    ks = record_keys(s1)
    # frequency filter + IDF weight, computed on the S2/S3 side
    df_count = ko.key.value_counts()
    ktype = np.array([k[0] for k in df_count.index])
    cap = np.array([caps[t] for t in ktype])
    keep = df_count.values <= cap
    df_count, ktype = df_count[keep], ktype[keep]
    N = len(other)
    w = np.log1p(N / df_count.values.astype(float)) * np.array([WEIGHT[t] for t in ktype])
    kw = pd.Series(w, index=np.asarray(df_count.index, dtype=object))
    codes = {k: j for j, k in enumerate(kw.index)}
    ko = ko[ko.key.isin(kw.index)]
    ks = ks[ks.key.isin(kw.index)]
    ko = pd.DataFrame({"code": ko.key.map(codes).values.astype(np.int32), "orow": ko.row.values})
    ks = pd.DataFrame({"code": ks.key.map(codes).values.astype(np.int32), "srow": ks.row.values})
    wts = kw.values.astype(np.float32)
    if verbose:
        print(f"  keys built: {len(ks):,} S1 keys, {len(ko):,} S2/S3 keys ({time.time()-t0:.0f}s)", flush=True)

    out = []
    s1_ids, o_ids = s1.entity_id.values, other.entity_id.values
    for start in range(0, len(s1), CHUNK):
        part = ks[(ks.srow >= start) & (ks.srow < start + CHUNK)]
        pairs = part.merge(ko, on="code")
        pairs["w"] = wts[pairs.code.values]
        sc = pairs.groupby(["srow", "orow"], sort=False)["w"].sum().reset_index()
        sc = sc.sort_values(["srow", "w"], ascending=[True, False])
        sc["rank"] = sc.groupby("srow").cumcount() + 1
        sc = sc[sc["rank"] <= k_max]
        out.append(pd.DataFrame({"s1_id": s1_ids[sc.srow.values], "cand_id": o_ids[sc.orow.values],
                                 "score": sc.w.values.astype(np.float32), "rank": sc["rank"].values.astype(np.int16)}))
        if verbose and (start // CHUNK) % 5 == 0:
            print(f"  {min(start+CHUNK, len(s1)):,}/{len(s1):,} S1 done ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def recall_report(cands, gt, s1_country):
    truth = gt.assign(cand_id=gt.matched_entity_ids.str.split(",")).explode("cand_id")
    truth = truth[truth.cand_id.notna() & (truth.cand_id != "")][["source1_entity_id", "cand_id"]]
    truth.columns = ["s1_id", "cand_id"]
    m = truth.merge(cands[["s1_id", "cand_id", "rank"]], on=["s1_id", "cand_id"], how="left")
    m["country"] = m.s1_id.map(s1_country)
    res = {}
    print(f"\n  true pairs: {len(m):,}")
    print(f"  {'K':>5} {'recall':>8} {'avg cands':>10}   by country")
    for k in [5, 10, 20, 50, 100]:
        if k > K_MAX:
            break
        hit = m["rank"].le(k)
        avg = cands[cands["rank"] <= k].groupby("s1_id").size().reindex(gt.source1_entity_id, fill_value=0).mean()
        byc = "  ".join(f"{c}={v*100:.1f}%" for c, v in hit.groupby(m.country).mean().items())
        print(f"  {k:>5} {hit.mean()*100:>7.2f}% {avg:>10.1f}   {byc}")
        res[k] = (hit.mean(), avg)
    missed = m[m["rank"].isna()]
    return res, missed


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "dev"
    s1 = pd.read_parquet(f"{PQ}/{prefix}_source1_norm.parquet")
    other = pd.concat([pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet") for k in "23"],
                      ignore_index=True)
    print(f"S1: {len(s1):,}   S2+S3: {len(other):,}")
    t = time.time()
    cands = generate(s1, other)
    print(f"blocking done in {time.time()-t:.0f}s")
    cands.to_parquet(f"{PQ}/{prefix}_candidates.parquet", index=False)

    gt = pd.read_parquet(f"{PQ}/{prefix}_ground_truth.parquet")
    res, missed = recall_report(cands, gt, s1.set_index("entity_id").country)

    # show missed true pairs so we can see WHY blocking misses them
    look = other.set_index("entity_id")[["business_name", "business_address"]]
    s1l = s1.set_index("entity_id")[["business_name", "business_address"]]
    print("\n  15 missed true pairs (S1  ->  missed S2/S3):")
    for _, r in missed.sample(min(15, len(missed)), random_state=0).iterrows():
        a, b = s1l.loc[r.s1_id], look.loc[r.cand_id]
        print(f"   S1 {a.business_name} | {a.business_address}\n   -> {b.business_name} | {b.business_address}\n")

    from explog import log
    k = 50 if 50 in res else max(res)
    log(f"blocking v2 (+name pairs, full name) K={k}", notes=f"{prefix}; caps={CAPS}",
        blocking_recall=res[k][0], avg_candidates=round(res[k][1], 1))
