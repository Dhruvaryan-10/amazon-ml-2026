"""Train the pair model on dev candidates and score macro F0.5 on the held-out dev entities.

Usage: python src/train.py dev
"""
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from features import pair_features
from metrics import macro_f05
from explog import log

PQ = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SEED = 42
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, verbose=-1, num_threads=4)


def decide(pairs, prob, thr, one_to_one=True):
    """pairs: s1_id, cand_id. Returns {s1_id: set(matches)}."""
    d = pairs[["s1_id", "cand_id"]].assign(p=prob)
    if one_to_one:   # each S2/S3 record goes only to the S1 entity it scores highest with
        d = d[d.p == d.groupby("cand_id").p.transform("max")]
    d = d[d.p >= thr]
    return d.groupby("s1_id").cand_id.agg(set).to_dict()


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "dev"
    t0 = time.time()
    s1 = pd.read_parquet(f"{PQ}/{prefix}_source1_norm.parquet")
    other = pd.concat([pd.read_parquet(f"{PQ}/{prefix}_source{k}_norm.parquet") for k in "23"],
                      ignore_index=True)
    cands = pd.read_parquet(f"{PQ}/{prefix}_candidates.parquet")
    gt = pd.read_parquet(f"{PQ}/{prefix}_ground_truth.parquet")
    truth = {r.source1_entity_id: set(r.matched_entity_ids.split(",")) if r.matched_entity_ids else set()
             for r in gt.itertuples()}
    true_pairs = {(s, c) for s, cs in truth.items() for c in cs}
    print(f"loaded: {len(cands):,} candidate pairs ({time.time()-t0:.0f}s)", flush=True)

    X = pair_features(cands, s1, other)
    y = np.array([(s, c) in true_pairs for s, c in zip(cands.s1_id.values, cands.cand_id.values)], dtype=np.int8)
    print(f"features: {X.shape[1]} columns ({time.time()-t0:.0f}s); positive rate {y.mean()*100:.1f}%", flush=True)

    is_val = cands.s1_id.map(s1.set_index("entity_id").is_val).values.astype(bool)
    dtr = lgb.Dataset(X[~is_val], y[~is_val])
    dva = lgb.Dataset(X[is_val], y[is_val])
    model = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    p = model.predict(X[is_val], num_iteration=model.best_iteration)
    print(f"model trained: {model.best_iteration} trees ({time.time()-t0:.0f}s)", flush=True)

    # score on ALL held-out dev S1 entities (incl. those blocking found nothing for)
    val_ids = set(s1.entity_id[s1.is_val])
    truth_val = {s: truth[s] for s in val_ids}
    vp = cands[is_val].reset_index(drop=True)
    print("\n  threshold   F0.5 (no 1:1)   F0.5 (1:1 ownership)")
    best = (0, None)
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        f_plain = macro_f05(decide(vp, p, thr, one_to_one=False), truth_val)
        f_own = macro_f05(decide(vp, p, thr, one_to_one=True), truth_val)
        print(f"  {thr:>9.2f}   {f_plain:>13.4f}   {f_own:>20.4f}")
        if f_own > best[0]:
            best = (f_own, thr)
    print(f"\nbest: F0.5 = {best[0]:.4f} at threshold {best[1]} (1:1 ownership)")

    imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("\ntop 15 features (gain):")
    print((imp / imp.sum() * 100).round(1).head(15).to_string())

    os.makedirs(OUT, exist_ok=True)
    model.save_model(os.path.join(OUT, f"lgb_{prefix}.txt"))
    log("LGBM v1 + 1:1 ownership + threshold", notes=f"{prefix} val; thr={best[1]}; trees={model.best_iteration}",
        cv_f05=best[0])
