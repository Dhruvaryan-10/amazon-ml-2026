"""Candidate generation (blocking) by weighted key overlap - memory-lean numpy version.

Every record emits "keys". Two records sharing a rare key are likely the same business:
  n  name token (or its phonetic skeleton) within the same country+state
  N  name token within the country (catches missing/wrong state) - stricter frequency cap
  p  first 5 letters of the space-less name, same country+state ("southerneducational" ~ "southern ...")
  a  house number + address word, same country ("1500|jupiter")
  P  PAIR of name tokens, whole country ("bengaluru+infra"): works with no state / no address
  e  the full space-less name, whole country ("bengaluruinfra")
  Q  pair of 4-letter token PREFIXES, whole country ("robe+glad"): survives late typos
  x  NAME token x ADDRESS word, whole country ("prm|gorakhpur", "lf|balkishunganj"): a common
     name word ("prime", "life") is rare once tied to a locality; the phonetic skeleton makes it
     work across scripts ("Prime" / "प्राइम" -> "prm")
(v4: dropped country-wide single-token "N" keys: at full density 99% of them exceed the cap)
Keys are hashed to int64. Keys shared by more than CAP[type] S2/S3 records are dropped.
Each S1 entity gets its top-K S2/S3 records by the sum of IDF weights of shared keys.

Usage:  python src/blocking.py dev          (dev sample, prints recall)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
PQ = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")

TYPES = ["n", "p", "a", "P", "e", "Q", "x"]
CAPS = {"n": 300, "p": 150, "a": 150, "P": 150, "e": 150, "Q": 100, "x": 150}   # dev scale
WEIGHT = {"n": 1.0, "p": 0.8, "a": 1.2, "P": 1.0, "e": 1.0, "Q": 0.7, "x": 1.2}
GENERIC_ADDR = set("""road street avenue drive lane nagar colony floor house block sector near opposite main
building apartment unit suite plot flat shop ground first second third city district west east north south
circle court place highway boulevard trail parkway village town area market cross phase park""".split())
K_MAX = 50
KEY_COLS = ["entity_id", "country", "state", "name_core", "name_cons", "name_nospace", "addr_nums", "addr_words"]


def _keys_of(c, st, core, cons, nospace, nums, words):
    ks = set()
    for t in set(core.split()) | set(cons.split()):
        if len(t) >= 3 or (t.isdigit() and len(t) >= 2):
            ks.add(f"n|{c}|{st}|{t}")
    if len(nospace) >= 5:
        ks.add(f"p|{c}|{st}|{nospace[:5]}")
    if len(nospace) >= 4:
        ks.add(f"e|{c}|{nospace}")
    for toks in (core.split(), cons.split()):
        toks = sorted({t for t in toks if len(t) >= 2})[:8]
        for x in range(len(toks)):
            for y in range(x + 1, len(toks)):
                ks.add(f"P|{c}|{toks[x]}|{toks[y]}")
    pre = sorted({t[:4] for t in core.split() if len(t) >= 4})[:8]
    for x in range(len(pre)):
        for y in range(x + 1, len(pre)):
            ks.add(f"Q|{c}|{pre[x]}|{pre[y]}")
    ws = [w for w in words.split() if len(w) >= 4]
    # name x address cross keys: 3 longest name tokens (words + skeletons) x 4 longest locality words
    loc = sorted({w for w in ws if w not in GENERIC_ADDR}, key=len, reverse=True)[:4]
    if loc:
        nt = sorted({t for t in core.split() if len(t) >= 3}, key=len, reverse=True)[:3]
        nt += sorted({t for t in cons.split() if len(t) >= 2}, key=len, reverse=True)[:3]
        for t in set(nt):
            for w in loc:
                ks.add(f"x|{c}|{t}|{w}")
    for n in nums.split():
        if n != "0":
            for w in ws:
                ks.add(f"a|{c}|{n}|{w}")
    return ks


def record_keys(df, batch=200_000):
    """-> (row int32, hash int64, type int8) arrays for every key of every record."""
    tcode = {t: i for i, t in enumerate(TYPES)}
    R, H, T = [], [], []
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        rows, hs, ts = [], [], []
        cols = [part[c].tolist() for c in KEY_COLS[1:]]
        for i, vals in enumerate(zip(*cols)):
            for k in _keys_of(*vals):
                rows.append(start + i)
                hs.append(hash(k))
                ts.append(tcode[k[0]])
        R.append(np.array(rows, dtype=np.int32)); H.append(np.array(hs, dtype=np.int64))
        T.append(np.array(ts, dtype=np.int8))
    return np.concatenate(R), np.concatenate(H), np.concatenate(T)


class KeyIndex:
    """Index over the S2/S3 pool; query it with S1 chunks."""

    def __init__(self, other, caps=CAPS, verbose=True):
        t0 = time.time()
        self.ids = other.entity_id.to_numpy()
        rows, h, t = record_keys(other)
        order = np.argsort(h, kind="stable")
        h.sort(kind="stable")                  # in place: avoids a second 8-byte copy of every key
        rows = rows[order]
        t = t[order]
        del order
        # h is sorted, so distinct keys are just the run boundaries (no extra sort / memory)
        start = np.r_[0, np.flatnonzero(np.diff(h)) + 1]
        count = np.diff(np.r_[start, len(h)])
        self.uniq = h[start]
        del h
        utype = t[start]
        del t
        self.start, self.count = start.astype(np.int64), count.astype(np.int32)
        cap = np.array([caps[x] for x in TYPES])[utype]
        w = np.array([WEIGHT[x] for x in TYPES])[utype]
        self.keep = self.count <= cap
        self.weight = (np.log1p(len(other) / self.count) * w).astype(np.float32)
        self.rows = rows
        if verbose:
            print(f"  index: {len(self.rows):,} keys, {len(self.uniq):,} distinct, {self.keep.mean()*100:.1f}% kept "
                  f"({time.time()-t0:.0f}s)", flush=True)

    def query(self, s1, k_max=K_MAX, chunk=5000, verbose=True, as_rows=False):
        t0 = time.time()
        srow_all, sh_all, _ = record_keys(s1)
        idx = np.searchsorted(self.uniq, sh_all)
        idx[idx >= len(self.uniq)] = 0
        ok = (self.uniq[idx] == sh_all) & self.keep[idx]
        srow_all, idx = srow_all[ok], idx[ok]
        s1_ids = s1.entity_id.to_numpy()
        N = np.int64(len(self.ids))
        out = []
        bounds = np.searchsorted(srow_all, np.arange(0, len(s1) + chunk, chunk))  # srow_all is sorted
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], bounds[b + 1]
            if lo == hi:
                continue
            sr, ix = srow_all[lo:hi], idx[lo:hi]
            lens = self.count[ix].astype(np.int64)
            tot = int(lens.sum())
            offs = np.repeat(self.start[ix] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(tot)
            orow = self.rows[offs]
            pid = np.repeat(sr.astype(np.int64), lens) * N + orow
            w = np.repeat(self.weight[ix], lens)
            sc = pd.Series(w).groupby(pid).sum()
            pid, score = sc.index.to_numpy(), sc.to_numpy().astype(np.float32)
            srow, orw = pid // N, pid % N
            o = np.lexsort((-score, srow))
            srow, orw, score = srow[o], orw[o], score[o]
            first = np.r_[0, np.flatnonzero(np.diff(srow)) + 1]
            rank = np.arange(len(srow)) - np.repeat(first, np.diff(np.r_[first, len(srow)])) + 1
            keep = rank <= k_max
            if as_rows:
                out.append(pd.DataFrame({"s1_row": srow[keep].astype(np.int32), "o_row": orw[keep].astype(np.int32),
                                         "score": score[keep], "rank": rank[keep].astype(np.int16)}))
            else:
                out.append(pd.DataFrame({"s1_id": s1_ids[srow[keep]], "cand_id": self.ids[orw[keep]],
                                         "score": score[keep], "rank": rank[keep].astype(np.int16)}))
            if verbose and b % 20 == 0:
                print(f"    {min((b+1)*chunk, len(s1)):,}/{len(s1):,} S1 queried ({time.time()-t0:.0f}s)", flush=True)
        if not out:
            return pd.DataFrame(columns=["s1_id", "cand_id", "score", "rank"])
        return pd.concat(out, ignore_index=True)


def generate(s1, other, caps=CAPS, k_max=K_MAX, verbose=True):
    return KeyIndex(other, caps, verbose).query(s1, k_max, verbose=verbose)


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
        if k > cands["rank"].max():
            break
        hit = m["rank"].le(k)
        avg = cands[cands["rank"] <= k].groupby("s1_id").size().reindex(gt.source1_entity_id, fill_value=0).mean()
        byc = "  ".join(f"{c}={v*100:.1f}%" for c, v in hit.groupby(m.country).mean().items())
        print(f"  {k:>5} {hit.mean()*100:>7.2f}% {avg:>10.1f}   {byc}")
        res[k] = (hit.mean(), avg)
    return res, m[m["rank"].isna()]


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
    look = other.set_index("entity_id")[["business_name", "business_address"]]
    s1l = s1.set_index("entity_id")[["business_name", "business_address"]]
    print("\n  10 missed true pairs (S1  ->  missed S2/S3):")
    for _, r in missed.sample(min(10, len(missed)), random_state=0).iterrows():
        a, b = s1l.loc[r.s1_id], look.loc[r.cand_id]
        print(f"   S1 {a.business_name} | {a.business_address}\n   -> {b.business_name} | {b.business_address}\n")
    from explog import log
    k = 50 if 50 in res else max(res)
    log(f"blocking v4 (+name x address, -N) K={k}", notes=f"{prefix}; caps={CAPS}",
        blocking_recall=res[k][0], avg_candidates=round(res[k][1], 1))
