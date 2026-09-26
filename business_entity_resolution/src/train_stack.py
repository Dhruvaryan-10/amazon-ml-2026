"""Stage-2 model on top of the stage-1 LightGBM. Uses the features saved by `pipeline.py train`.
  1. stage-1 out-of-fold probabilities on the train entities (5 folds by entity, no leakage)
  2. stage-2 features (stack.py) + stage-1 features -> stage-2 LightGBM
  3. macro F0.5 on the 25k val entities, compared with stage-1 alone; error examples
Usage: python src/train_stack.py        (~30-40 min)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from metrics import macro_f05
from train_full import decide, PARAMS, PQ, OUT, SEED
from stack import stack_features
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


if __name__ == "__main__":
    meta = json.load(open(os.path.join(OUT, "lgb_full.json")))
    F1 = meta["features"]
    tr = pd.read_parquet(f"{PQ}/full_train_feats.parquet").sort_values(["s1_id"], kind="stable").reset_index(drop=True)
    va = pd.read_parquet(f"{PQ}/full_val_feats.parquet").sort_values(["s1_id"], kind="stable").reset_index(drop=True)
    say(f"loaded train {len(tr):,} / val {len(va):,} pairs")

    # 1. stage-1 OOF on train
    ents = tr.s1_id.unique()
    fold_of = pd.Series(np.random.RandomState(SEED).randint(0, 5, len(ents)), index=ents)
    fold = tr.s1_id.map(fold_of).to_numpy()
    p1_tr = np.zeros(len(tr))
    rounds = int(meta.get("trees", 1200))
    for f in range(5):
        m = lgb.train(PARAMS, lgb.Dataset(tr.loc[fold != f, F1], tr.label[fold != f]), num_boost_round=rounds)
        p1_tr[fold == f] = m.predict(tr.loc[fold == f, F1])
        say(f"stage-1 fold {f+1}/5 done")
    m1 = lgb.Booster(model_file=os.path.join(OUT, "lgb_full.txt"))
    p1_va = m1.predict(va[F1])

    # 2. stage-2 features
    recs = load_recs("train", set(tr.cand_id) | set(va.cand_id))
    say(f"records loaded ({len(recs):,})")
    S_tr = stack_features(tr, p1_tr, recs); say("stack features train done")
    S_va = stack_features(va, p1_va, recs); say("stack features val done")
    X_tr = pd.concat([tr[F1].reset_index(drop=True), S_tr.reset_index(drop=True)], axis=1)
    X_va = pd.concat([va[F1].reset_index(drop=True), S_va.reset_index(drop=True)], axis=1)
    F2 = list(X_tr.columns)
    es = np.isin(tr.s1_id.to_numpy(), np.random.RandomState(SEED + 1).choice(ents, len(ents) // 10, replace=False))
    m2 = lgb.train(PARAMS, lgb.Dataset(X_tr[~es], tr.label[~es]), num_boost_round=3000,
                   valid_sets=[lgb.Dataset(X_tr[es], tr.label[es])],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    p2_va = m2.predict(X_va, num_iteration=m2.best_iteration)
    say(f"stage-2 trained ({m2.best_iteration} trees)")

    # 3. compare
    ids = pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(ids))]
    truth = {s: set(x.split(",")) if x else set() for s, x in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    print("\n  threshold   stage-1   stage-2")
    res1, res2 = {}, {}
    for thr in np.round(np.arange(0.4, 0.91, 0.05), 2):
        res1[thr] = macro_f05(decide(va, p1_va, thr), truth)
        res2[thr] = macro_f05(decide(va, p2_va, thr), truth)
        print(f"  {thr:>9.2f}   {res1[thr]:.4f}    {res2[thr]:.4f}")
    t1, t2 = max(res1, key=res1.get), max(res2, key=res2.get)
    print(f"\nstage-1 best {res1[t1]:.4f} (thr {t1})   |   STAGE-2 best {res2[t2]:.4f} (thr {t2})")
    imp = pd.Series(m2.feature_importance("gain"), index=F2).sort_values(ascending=False)
    print("\ntop 12 stage-2 features:\n" + (imp / imp.sum() * 100).round(1).head(12).to_string())

    # error examples with the better model
    pred = decide(va, p2_va, t2)
    s1r = pd.read_parquet(f"{PQ}/train_source1.parquet").set_index("entity_id")
    orr = pd.concat([pd.read_parquet(f"{PQ}/train_source{k}.parquet") for k in "23"]).set_index("entity_id")
    fp = [(s, c) for s, cs in pred.items() for c in cs - truth.get(s, set())]
    fn = [(s, c) for s, cs in truth.items() for c in cs - pred.get(s, set()) if c in set(va.cand_id)]
    rng = np.random.RandomState(0)
    for name, lst in [("FALSE MATCHES (we said match, truth says no)", fp),
                      ("MISSED MATCHES (in candidates, model said no)", fn)]:
        print(f"\n{name}: {len(lst):,} total; 10 examples")
        for k in rng.choice(len(lst), min(10, len(lst)), replace=False):
            s, c = lst[k]
            print(f"  S1 {s1r.loc[s,'business_name']} | {s1r.loc[s,'business_address']}")
            print(f"  -> {orr.loc[c,'business_name']} | {orr.loc[c,'business_address']}\n")

    m2.save_model(os.path.join(OUT, "lgb_stack.txt"), num_iteration=m2.best_iteration)
    json.dump({"f1": F1, "f2": F2, "threshold": t2, "use_stack": bool(res2[t2] > res1[t1] + 0.0005)},
              open(os.path.join(OUT, "lgb_stack.json"), "w"))
    log("stage-2 stacking (S2<->S3 agreement)", notes=f"stage1={res1[t1]:.4f}; thr={t2}", cv_f05=res2[t2])
