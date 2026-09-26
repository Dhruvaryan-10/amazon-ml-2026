"""Stage-2 ("stacking") features built from stage-1 probabilities of ALL candidates of the same S1:
  - where this pair sits among its S1's candidates (rank, gap to best, how many confident ones)
  - S2<->S3 agreement: does this candidate look like the OTHER confident candidates of the same S1?
    (S1 in English, S2 and S3 both in Hindi -> S2 and S3 resemble each other more than S1)
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

MUT_MIN_P1 = 0.02     # only compute agreement for pairs stage-1 doesn't already reject
CONF_P1 = 0.3         # "confident" other candidates


def stack_features(pairs, p1, recs):
    """pairs: DataFrame(s1_id, cand_id) grouped by s1_id (contiguous); p1: stage-1 probs;
    recs: DataFrame indexed by cand entity_id with name_cons, name_nospace, addr_norm, addr_nums."""
    d = pairs[["s1_id", "cand_id"]].copy()
    d["p1"] = p1
    g = d.groupby("s1_id").p1
    d["p1_rank"] = g.rank(ascending=False, method="first")
    d["p1_max"] = g.transform("max")
    d["p1_gap"] = d.p1_max - d.p1
    d["p1_sum"] = g.transform("sum")
    d["p1_n_conf"] = g.transform(lambda s: (s >= 0.5).sum())
    d["p1_second"] = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
    d["is_s3"] = d.cand_id.str.startswith("S3-").astype(np.int8)
    # same-source competition: rank among candidates from the same source
    d["p1_rank_src"] = d.groupby(["s1_id", "is_s3"]).p1.rank(ascending=False, method="first")

    # S2<->S3 agreement with the other confident candidates of the same S1
    nm = np.full(len(d), np.nan, dtype=np.float32)
    ad = np.full(len(d), np.nan, dtype=np.float32)
    ns = np.full(len(d), np.nan, dtype=np.float32)
    wsum = np.zeros(len(d), dtype=np.float32)
    cons = recs.name_cons.to_dict(); addr = recs.addr_norm.to_dict(); nosp = recs.name_nospace.to_dict()
    cand = d.cand_id.to_numpy(); pp = d.p1.to_numpy(); s1 = d.s1_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, len(s1)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = np.arange(a, b)
        conf = idx[pp[idx] >= CONF_P1]
        if len(conf) == 0:
            continue
        for i in idx[pp[idx] >= MUT_MIN_P1]:
            others = conf[conf != i]
            if len(others) == 0:
                continue
            ci = cand[i]
            best_n = best_a = best_s = 0.0
            w = 0.0
            for j in others:
                cj = cand[j]
                sn = fuzz.token_set_ratio(cons[ci], cons[cj])
                sa = fuzz.token_set_ratio(addr[ci], addr[cj]) if addr[ci] and addr[cj] else 0.0
                ss = fuzz.ratio(nosp[ci], nosp[cj])
                best_n, best_a, best_s = max(best_n, sn), max(best_a, sa), max(best_s, ss)
                w += pp[j] * max(sn, sa) / 100.0
            nm[i], ad[i], ns[i], wsum[i] = best_n, best_a, best_s, w
    d["mut_name"], d["mut_addr"], d["mut_nospace"], d["mut_weighted"] = nm, ad, ns, wsum
    return d.drop(columns=["s1_id", "cand_id", "is_s3"])
