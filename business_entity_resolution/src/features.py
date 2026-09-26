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


def pair_features(cands, s1, other):
    """cands: DataFrame(s1_id, cand_id, score, rank). s1/other: normalised frames."""
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
        "ad_tset", "ad_ratio", "ad_words_jacc", "ad_nums_jacc", "ad_nums_overlap"]}
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
