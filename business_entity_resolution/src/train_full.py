"""Train LightGBM on full-density features and measure macro F0.5 on the 25k held-out val entities.

Usage: python src/train_full.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from metrics import macro_f05
from explog import log

ROOT = os.path.join(os.path.dirname(__file__), "..")
PQ, OUT = os.path.join(ROOT, "data", "parquet"), os.path.join(ROOT, "outputs")
SEED = 42
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=4)


def decide(pairs, prob, thr):
    """1:1 ownership (each S2/S3 record -> its best S1) then threshold. -> {s1_id: set}"""
    d = pairs[["s1_id", "cand_id"]].assign(p=prob)
    d = d[d.p == d.groupby("cand_id").p.transform("max")]
    d = d[d.p >= thr]
    return d.groupby("s1_id").cand_id.agg(set).to_dict()


if __name__ == "__main__":
    t0 = time.time()
    tr = pd.read_parquet(f"{PQ}/full_train_feats.parquet")
    va = pd.read_parquet(f"{PQ}/full_val_feats.parquet")
    if "is_comp" in tr:              # stage-1 trains/evaluates on the sampled entities' own pairs
        tr = tr[tr.is_comp == 0].reset_index(drop=True)
        va = va[va.is_comp == 0].reset_index(drop=True)
    feats = [c for c in tr.columns if c not in ("s1_id", "cand_id", "label", "is_comp")]
    # early-stopping set: 10% of TRAIN entities (never the val entities we report on)
    ents = tr.s1_id.unique()
    es_ents = set(np.random.RandomState(SEED).choice(ents, len(ents) // 10, replace=False))
    es = tr.s1_id.isin(es_ents).to_numpy()
    model = lgb.train(PARAMS, lgb.Dataset(tr.loc[~es, feats], tr.label[~es]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(tr.loc[es, feats], tr.label[es])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    print(f"trained {model.best_iteration} trees ({time.time()-t0:.0f}s)")
    p = model.predict(va[feats], num_iteration=model.best_iteration)

    ids = pd.read_parquet(f"{PQ}/full_val_ids.parquet").entity_id
    gt = pd.read_parquet(f"{PQ}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(set(ids))]
    truth = {s: set(m.split(",")) if m else set() for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    print("\n  threshold   F0.5")
    res = {}
    for thr in np.arange(0.3, 0.96, 0.05):
        res[round(thr, 2)] = macro_f05(decide(va, p, thr), truth)
        print(f"  {thr:>9.2f}   {res[round(thr, 2)]:.4f}")
    thr = max(res, key=res.get)
    print(f"\nFULL-DENSITY val F0.5 = {res[thr]:.4f} at threshold {thr}")
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\ntop 10 features:\n" + (imp / imp.sum() * 100).round(1).head(10).to_string())
    model.save_model(os.path.join(OUT, "lgb_full.txt"), num_iteration=model.best_iteration)
    json.dump({"threshold": thr, "features": feats, "trees": model.best_iteration},
              open(os.path.join(OUT, "lgb_full.json"), "w"))
    log("LGBM full-density", notes=f"25k val entities; thr={thr}; trees={model.best_iteration}", cv_f05=res[thr])
