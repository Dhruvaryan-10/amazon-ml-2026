"""Compare decision rules on the saved full-density val predictions (~3 min, no re-blocking).
  A) global threshold (current)
  B) per-entity expected-F0.5 set selection: for each S1 choose the top-k (or empty set)
     that maximises expected F0.5 given the model's probabilities
Usage: python src/decide_eval.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from metrics import macro_f05
from train_full import decide, PQ, OUT
from explog import log


def expected_f_select(pairs, prob, miss_mass=0.0, min_p=0.05):
    """1:1 ownership, then per S1 pick k maximising E[F0.5] (ratio-of-expectations approximation).
    miss_mass: expected number of true matches blocking never found, added to every entity's truth count."""
    d = pairs[["s1_id", "cand_id"]].assign(p=prob)
    d = d[d.p == d.groupby("cand_id").p.transform("max")]
    d = d[d.p >= min_p].sort_values(["s1_id", "p"], ascending=[True, False])
    out = {}
    for s, g in d.groupby("s1_id", sort=False):
        p = g.p.to_numpy()
        T = p.sum() + miss_mass
        best_k, best = 0, np.prod(1 - p)              # E[F] of predicting nothing ~ P(no match)
        tp = np.cumsum(p)
        for k in range(1, len(p) + 1):
            f = 1.25 * tp[k-1] / (1.25 * tp[k-1] + 0.25 * (T - tp[k-1]) + (k - tp[k-1]))
            if f > best:
                best, best_k = f, k
        if best_k:
            out[s] = set(g.cand_id.to_numpy()[:best_k])
    return out


if __name__ == "__main__":
    meta = json.load(open(os.path.join(OUT, "lgb_full.json")))
    model = lgb.Booster(model_file=os.path.join(OUT, "lgb_full.txt"))
    va = pd.read_parquet(f"{PQ}/full_val_feats.parquet")
    p = model.predict(va[meta["features"]])
    ids = pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(ids))]
    truth = {s: set(m.split(",")) if m else set() for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    fa = macro_f05(decide(va, p, meta["threshold"]), truth)
    print(f"A) global threshold {meta['threshold']}: F0.5 = {fa:.4f}")
    best = (0, None)
    for mm in [0.0, 0.1, 0.2, 0.3]:
        fb = macro_f05(expected_f_select(va, p, miss_mass=mm), truth)
        print(f"B) expected-F selection, miss_mass={mm}: F0.5 = {fb:.4f}")
        best = max(best, (fb, mm))
    # where are the errors? (with the better rule)
    pred = expected_f_select(va, p, miss_mass=best[1]) if best[0] > fa else decide(va, p, meta["threshold"])
    rows = []
    for s, t in truth.items():
        q = pred.get(s, set())
        rows.append((len(t) == 0, len(q) == 0, len(q & t), len(q - t), len(t - q)))
    r = pd.DataFrame(rows, columns=["true_empty", "pred_empty", "tp", "fp", "fn"])
    print(f"\nerror breakdown ({len(r):,} val entities):")
    print(f"  singletons correctly left empty : {(r.true_empty & r.pred_empty).sum():,} / {r.true_empty.sum():,}")
    print(f"  entities with >=1 false match   : {(r.fp > 0).sum():,}  ({(r.fp>0).mean()*100:.1f}%)")
    print(f"  entities with >=1 missed match  : {(r.fn > 0).sum():,}  ({(r.fn>0).mean()*100:.1f}%)")
    print(f"  total FP {r.fp.sum():,}   total FN {r.fn.sum():,}   total TP {r.tp.sum():,}")
    json.dump({**meta, "decision": "expected_f" if best[0] > fa else "threshold", "miss_mass": best[1]},
              open(os.path.join(OUT, "lgb_full.json"), "w"))
    log("decision: expected-F vs threshold", notes=f"threshold={fa:.4f}; expF(miss={best[1]})={best[0]:.4f}",
        cv_f05=max(fa, best[0]))
