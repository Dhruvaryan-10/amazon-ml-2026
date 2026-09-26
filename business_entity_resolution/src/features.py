"""Pair features for (S1 entity, candidate) pairs. Country-agnostic: no country one-hot."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

NAME_COLS = ["name_norm", "name_core", "name_cons", "name_nospace", "legal", "name_is_domain"]
ADDR_COLS = ["addr_norm", "addr_words", "addr_nums", "postcode", "state"]


def _jacc(a, b):
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def _tok_match(t, others):
    """t matches some token in others exactly, by phonetic skeleton or by >=85 fuzzy ratio (typos)."""
    if t in others:
        return True
    return any(fuzz.ratio(t, o) >= 85 for o in others)


def pair_features(cands, s1, other, tokfreq=None, n_pool=1):
    """cands: DataFrame(s1_id, cand_id, score, rank). s1/other: normalised frames.
    tokfreq: {name token: #pool records containing it} -> how unusual a differing word is."""
    cols = ["entity_id", "business_name"] + NAME_COLS + ADDR_COLS
    A = s1[cols].drop_duplicates("entity_id").set_index("entity_id").loc[cands.s1_id.values]
    B = other[cols].drop_duplicates("entity_id").set_index("entity_id").loc[cands.cand_id.values]
    n = len(cands)
    F = {}

    # ---- blocking / context features (very strong) ----
    F["blk_score"] = cands.score.values
    F["blk_rank"] = cands["rank"].values
    g = cands.groupby("s1_id").score
    F["blk_score_rel"] = cands.score.values / g.transform("max").values
    F["blk_gap_top"] = g.transform("max").values - cands.score.values
    F["blk_n_cands"] = g.transform("size").values
    # rank of this S1 among all S1s that picked this candidate (reciprocal best match).
    # At full scale these are precomputed over ALL S1 entities (see pipeline.py).
    if "cand_rank" in cands:
        F["blk_cand_rank"] = cands.cand_rank.values
        F["blk_cand_n_s1"] = cands.cand_n_s1.values
    else:
        F["blk_cand_rank"] = cands.groupby("cand_id").score.rank(ascending=False, method="min").values
        F["blk_cand_n_s1"] = cands.groupby("cand_id").score.transform("size").values
    F["is_s3"] = cands.cand_id.str.startswith("S3-").values.astype(np.int8)

    # ---- string similarities ----
    out = {k: np.empty(n, dtype=np.float32) for k in [
        "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_cons_ratio", "nm_cons_tset",
        "nm_nospace_ratio", "nm_raw_tset", "nm_tok_jacc", "nm_len_diff",
        "ad_tset", "ad_ratio", "ad_words_jacc", "ad_nums_jacc", "ad_nums_overlap",
        "ad_num_variant", "nm_nospace_partial",
        "nm_a_only_n", "nm_b_only_n", "nm_a_only_maxidf", "nm_b_only_maxidf", "nm_a_only_sumidf",
        "nm_b_only_sumidf", "nm_b_rare_single", "nm_min_idf_shared", "ad_num_absdiff", "legal_jacc"]}
    tf = tokfreq or {}
    def idf(t):
        return float(np.log1p(n_pool / (1 + tf.get(t, 0))))
    a_leg, b_leg = A.legal.values, B.legal.values
    a_core, b_core = A.name_core.values, B.name_core.values
    a_cons, b_cons = A.name_cons.values, B.name_cons.values
    a_ns, b_ns = A.name_nospace.values, B.name_nospace.values
    a_raw, b_raw = A.business_name.values, B.business_name.values
    a_ad, b_ad = A.addr_norm.values, B.addr_norm.values
    a_w, b_w = A.addr_words.values, B.addr_words.values
    a_n, b_n = A.addr_nums.values, B.addr_nums.values
    for i in range(n):
        x, y = a_core[i], b_core[i]
        out["nm_ratio"][i] = fuzz.ratio(x, y)
        out["nm_tset"][i] = fuzz.token_set_ratio(x, y)
        out["nm_tsort"][i] = fuzz.token_sort_ratio(x, y)
        out["nm_partial"][i] = fuzz.partial_ratio(x, y)
        out["nm_jw"][i] = JaroWinkler.similarity(x, y)
        out["nm_cons_ratio"][i] = fuzz.ratio(a_cons[i], b_cons[i])
        out["nm_cons_tset"][i] = fuzz.token_set_ratio(a_cons[i], b_cons[i])
        out["nm_nospace_ratio"][i] = fuzz.ratio(a_ns[i], b_ns[i])
        out["nm_raw_tset"][i] = fuzz.token_set_ratio(a_raw[i].lower(), b_raw[i].lower())
        ta, tb = set(x.split()), set(y.split())
        out["nm_tok_jacc"][i] = _jacc(ta, tb)
        out["nm_len_diff"][i] = abs(len(x) - len(y))
        if a_ad[i] and b_ad[i]:
            out["ad_tset"][i] = fuzz.token_set_ratio(a_ad[i], b_ad[i])
            out["ad_ratio"][i] = fuzz.ratio(a_ad[i], b_ad[i])
        else:
            out["ad_tset"][i] = out["ad_ratio"][i] = np.nan
        out["ad_words_jacc"][i] = _jacc(set(a_w[i].split()), set(b_w[i].split()))
        na, nb = set(a_n[i].split()), set(b_n[i].split())
        out["ad_nums_jacc"][i] = _jacc(na, nb)
        out["ad_nums_overlap"][i] = len(na & nb)
        # house numbers equal after dropping a leading/trailing digit on either side (source noise)
        va = na | {n[1:] for n in na if len(n) >= 3} | {n[:-1] for n in na if len(n) >= 3}
        vb = nb | {n[1:] for n in nb if len(n) >= 3} | {n[:-1] for n in nb if len(n) >= 3}
        out["ad_num_variant"][i] = np.nan if not na or not nb else float(bool((va & nb) or (vb & na)))
        # "morrisheartlandsun" (domain) vs "morris heartland sun": containment of the space-less names
        out["nm_nospace_partial"][i] = fuzz.partial_ratio(a_ns[i], b_ns[i])
        # which name words differ, and how unusual they are (decoys swap one specific word:
        # "Eminent Plastic" vs "Eminent Beverages"; noise adds generic ones: "... Services")
        a_only = [t for t in ta if not _tok_match(t, tb)]
        b_only = [t for t in tb if not _tok_match(t, ta)]
        ia, ib = [idf(t) for t in a_only], [idf(t) for t in b_only]
        out["nm_a_only_n"][i], out["nm_b_only_n"][i] = len(a_only), len(b_only)
        out["nm_a_only_maxidf"][i] = max(ia) if ia else 0.0
        out["nm_b_only_maxidf"][i] = max(ib) if ib else 0.0
        out["nm_a_only_sumidf"][i], out["nm_b_only_sumidf"][i] = sum(ia), sum(ib)
        shared = ta & tb
        out["nm_min_idf_shared"][i] = min(idf(t) for t in shared) if shared else 0.0
        # a single made-up word nobody else uses ("Pyraveo") = renamed record -> trust the address
        out["nm_b_rare_single"][i] = float(len(tb) == 1 and tf.get(next(iter(tb)), 0) <= 2) if tb else 0.0
        di = [abs(int(x) - int(y)) for x in na for y in nb if len(x) < 9 and len(y) < 9]
        out["ad_num_absdiff"][i] = np.log1p(min(di)) if di else np.nan
        la, lb = set(a_leg[i].split()), set(b_leg[i].split())
        out["legal_jacc"][i] = _jacc(la, lb)
    F.update(out)

    # ---- categorical agreement: 1 = equal, 0 = conflict, nan = missing on a side ----
    def agree(a, b):
        a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        miss = (a == "") | (b == "")
        return np.where(miss, np.nan, (a == b).astype(float))
    F["legal_eq"] = agree(A.legal.values, B.legal.values)
    F["state_eq"] = agree(A.state.values, B.state.values)
    F["postcode_eq"] = agree(A.postcode.values, B.postcode.values)
    F["b_addr_empty"] = (B.addr_norm.values == "").astype(np.int8)
    F["b_is_domain"] = B.name_is_domain.values.astype(np.int8)
    F["b_non_latin"] = np.array([not s.isascii() for s in B.business_name.values], dtype=np.int8)
    F["a_name_len"] = np.array([len(s) for s in a_core], dtype=np.int16)
    F["b_name_len"] = np.array([len(s) for s in b_core], dtype=np.int16)
    F["a_n_tokens"] = np.array([s.count(" ") + 1 for s in a_core], dtype=np.int8)
    return pd.DataFrame(F)
