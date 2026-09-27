import sys, glob, pandas as pd
src = sys.argv[1]
s1_path = glob.glob("**/test_source1.tsv", recursive=True)[0]
s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], dtype=str)
m = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
fr = set(s1.loc[s1.country == "France", "entity_id"])
print("test S1:", len(s1), " France:", len(fr), " share:", round(len(fr) / len(s1), 4))
m["n"] = m.matched_entity_ids.str.len().gt(0) * (m.matched_entity_ids.str.count(",") + 1)
m = m.merge(s1, left_on="source1_entity_id", right_on="entity_id")
print(m.groupby("country").n.agg(["mean", lambda x: (x == 0).mean()]))
m.loc[m.country == "France", "matched_entity_ids"] = ""
m[["source1_entity_id", "matched_entity_ids"]].to_csv("outputs/probe_noFR.tsv", sep="\t", index=False)
print("wrote outputs/probe_noFR.tsv")