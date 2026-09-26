"""Stage-2 model: re-scores the pairs stage-1 finds plausible, using
  - stage-1 probability p1
  - agreement with the OTHER candidates of the same S1 (S2<->S3 agreement, rank, gap)   [stack_features]
  - COMPETITION with the other S1s that claim the same record, by probability          [comp_features]
Train entities use out-of-fold p1; competitor S1s (not sampled) use the stage-1 model, like test.
Usage: python src/train_stack.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from metrics import macro_f05
from train_full import decide, PARAMS, PQ, OUT, SEED
from stack import stack_features, comp_features, P_MIN, BLK_KEEP
from explog import log

T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def load_recs(prefix, ids):
    cols = ["entity_id", "name_cons", "name_nospace", "addr_norm"]
    parts = []
    for k in "23":
        t = pq.read_table(f"{PQ}/{prefix}_source{k}_norm.parquet", columns=cols)
        parts.append(t.filter(pc.is_in(t["entity_id"], value_set=pa.array(list(ids)))).to_pandas())
    return pd.concat(parts).set_index("entity_id")


def stage2_matrix(df, p1, recs):
    """df: own+competitor pairs (is_comp); p1 aligned. Returns (mask of own rows kept, X2)."""
    comp = comp_features(pd.factorize(df.cand_id)[0].astype(np.int64), p1.astype(np.float32))
    own = (df.is_comp == 0).to_numpy()
    o = df[own].reset_index(drop=True)
    S = stack_features(o, p1[own], recs)
    X2 = pd.concat([pd.DataFrame({"p1": p1[own]}), S.reset_index(drop=True),
                    pd.DataFrame({k: v[own] for k, v in comp.items()}),
                    o[[c for c in BLK_KEEP if c in o]].reset_index(drop=True)], axis=1)
    return own, X2


if __name__ == "__main__":
    meta = json.load(open(os.path.join(OUT, "lgb_full.json")))
    F1 = meta["features"]
    tr = pd.read_parquet(f"{PQ}/full_train_feats.parquet").sort_values(["is_comp", "s1_id"], kind="stable").reset_index(drop=True)
    va = pd.read_parquet(f"{PQ}/full_val_feats.parquet").sort_values(["is_comp", "s1_id"], kind="stable").reset_index(drop=True)
    say(f"loaded train {len(tr):,} / val {len(va):,} pairs (incl. competitor pairs)")
    m1 = lgb.Booster(model_file=os.path.join(OUT, "lgb_full.txt"))

    # stage-1 p1: out-of-fold for sampled train entities, stage-1 model for everything else
    own_tr = (tr.is_comp == 0).to_numpy()
    ents = tr.s1_id[own_tr].unique()
    fold_of = pd.Series(np.random.RandomState(SEED).randint(0, 5, len(ents)), index=ents)
    fold = tr.s1_id.map(fold_of).fillna(-1).to_numpy()
    p1_tr = m1.predict(tr[F1])
    rounds = int(meta.get("trees", 1200))
    for f in range(5):
        fit = own_tr & (fold != f)
        m = lgb.train(PARAMS, lgb.Dataset(tr.loc[fit, F1], tr.label[fit]), num_boost_round=rounds)
        p1_tr[fold == f] = m.predict(tr.loc[fold == f, F1])
        say(f"stage-1 fold {f+1}/5 done")
    p1_va = m1.predict(va[F1])

    recs = load_recs("train", set(tr.cand_id) | set(va.cand_id))
    say(f"records loaded ({len(recs):,})")
    own_t, X2_tr = stage2_matrix(tr, p1_tr, recs); say("stage-2 features train done")
    own_v, X2_va = stage2_matrix(va, p1_va, recs); say("stage-2 features val done")
    y_tr = tr.label[own_t].to_numpy()
    keep = X2_tr.p1.to_numpy() >= P_MIN
    F2 = list(X2_tr.columns)
    trs = tr[own_t].reset_index(drop=True)
    es = np.isin(trs.s1_id.to_numpy(), np.random.RandomState(SEED + 1).choice(ents, len(ents) // 10, replace=False))
    m2 = lgb.train(PARAMS, lgb.Dataset(X2_tr[keep & ~es], y_tr[keep & ~es]), num_boost_round=3000,
                   valid_sets=[lgb.Dataset(X2_tr[keep & es], y_tr[keep & es])],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    say(f"stage-2 trained ({m2.best_iteration} trees)")
    p1v = X2_va.p1.to_numpy()
    p2_va = p1v.copy()
    kv = p1v >= P_MIN
    p2_va[kv] = m2.predict(X2_va[kv], num_iteration=m2.best_iteration)

    vas = va[own_v].reset_index(drop=True)
    ids = pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(ids))]
    truth = {s: set(x.split(",")) if x else set() for s, x in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    print("\n  threshold   stage-1   stage-2")
    res1, res2 = {}, {}
    for thr in np.round(np.arange(0.4, 0.91, 0.05), 2):
        res1[thr] = macro_f05(decide(vas, p1v, thr), truth)
        res2[thr] = macro_f05(decide(vas, p2_va, thr), truth)
        print(f"  {thr:>9.2f}   {res1[thr]:.4f}    {res2[thr]:.4f}")
    t1, t2 = max(res1, key=res1.get), max(res2, key=res2.get)
    print(f"\nstage-1 best {res1[t1]:.4f} (thr {t1})   |   STAGE-2 best {res2[t2]:.4f} (thr {t2})")
    imp = pd.Series(m2.feature_importance("gain"), index=F2).sort_values(ascending=False)
    print("\ntop 12 stage-2 features:\n" + (imp / imp.sum() * 100).round(1).head(12).to_string())

    pred = decide(vas, p2_va, t2)
    s1r = pd.read_parquet(f"{PQ}/train_source1.parquet").set_index("entity_id")
    orr = pd.concat([pd.read_parquet(f"{PQ}/train_source{k}.parquet") for k in "23"]).set_index("entity_id")
    fp = [(s, c) for s, cs in pred.items() for c in cs - truth.get(s, set())]
    fn = [(s, c) for s, cs in truth.items() for c in cs - pred.get(s, set()) if c in set(vas.cand_id)]
    rng = np.random.RandomState(0)
    for name, lst in [("FALSE MATCHES", fp), ("MISSED MATCHES (in candidates)", fn)]:
        print(f"\n{name}: {len(lst):,} total; 8 examples")
        for k in rng.choice(len(lst), min(8, len(lst)), replace=False):
            s, c = lst[k]
            print(f"  S1 {s1r.loc[s,'business_name']} | {s1r.loc[s,'business_address']}")
            print(f"  -> {orr.loc[c,'business_name']} | {orr.loc[c,'business_address']}\n")

    m2.save_model(os.path.join(OUT, "lgb_stack.txt"), num_iteration=m2.best_iteration)
    json.dump({"f1": F1, "f2": F2, "p_min": P_MIN, "threshold": t2, "use_stack": bool(res2[t2] > res1[t1] + 0.0005)},
              open(os.path.join(OUT, "lgb_stack.json"), "w"))
    log("stage-2 v2: +S1-vs-S1 competition by probability", notes=f"stage1={res1[t1]:.4f}; thr={t2}", cv_f05=res2[t2])
