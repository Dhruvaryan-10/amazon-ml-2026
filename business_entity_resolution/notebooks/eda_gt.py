import os
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "train")
rd = lambda f, cols=None: pd.read_csv(f"{D}/{f}", sep="\t", dtype=str,
                                      keep_default_na=False, usecols=cols)

gt = rd("train_ground_truth.tsv")
s1 = rd("train_source1.tsv", ["entity_id", "country"])
gt = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")

gt["ids"] = gt["matched_entity_ids"].apply(lambda x: x.split(",") if x else [])
gt["n"]   = gt["ids"].str.len()
gt["n2"]  = gt["ids"].apply(lambda l: sum(i.startswith("S2-") for i in l))
gt["n3"]  = gt["ids"].apply(lambda l: sum(i.startswith("S3-") for i in l))

print("Q1. Singleton % (= score if you predict nothing):")
print("   overall:", round((gt.n == 0).mean() * 100, 2))
print(gt.groupby("country").apply(lambda g: round((g.n == 0).mean() * 100, 2)))

print("\nQ2. Matches per S1 entity (total, % of entities):")
print((gt.n.clip(upper=10).value_counts(normalize=True).sort_index() * 100).round(2))
print("\n   S2 matches per entity:\n", (gt.n2.clip(upper=6).value_counts(normalize=True).sort_index() * 100).round(2))
print("\n   S3 matches per entity:\n", (gt.n3.clip(upper=6).value_counts(normalize=True).sort_index() * 100).round(2))
print("\n   mean matches by country:\n", gt.groupby("country")[["n", "n2", "n3"]].mean().round(2))

ex = gt[["source1_entity_id", "ids"]].explode("ids").dropna()
dup = ex["ids"].value_counts()
print("\nQ3. S2/S3 ids that appear under >1 S1 entity:", int((dup > 1).sum()),
      "out of", len(dup))

for f in ["train_source2.tsv", "train_source3.tsv"]:
    ids = rd(f, ["entity_id"])["entity_id"]
    print(f"\nQ4. {f}: {len(ids)} records, "
          f"{round(ids.isin(dup.index).mean() * 100, 2)}% are a true match for some S1 "
          f"(the rest are pure distractors)")

print("\nQ5. Examples: 3 India + 3 US S1 entities with their matches")
S = {k: rd(f"train_source{k}.tsv").set_index("entity_id") for k in ["1", "2", "3"]}
for c in ["India", "US"]:
    for _, row in gt[(gt.country == c) & (gt.n >= 2)].sample(3, random_state=1).iterrows():
        print("\n  S1 |", " | ".join(S["1"].loc[row.source1_entity_id, ["business_name", "business_address"]]))
        for i in row.ids:
            src = S[i[1]]
            if i in src.index:
                print(f"  {i[:2]} |", " | ".join(src.loc[i, ["business_name", "business_address"]]))