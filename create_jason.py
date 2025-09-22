import json, pandas as pd
uids = pd.read_csv("kiuisobj_v1_merged_80K.csv", header=None)[1].tolist()
mapping = json.load(open("extensions/assets/gobjaverse_280k_index_to_objaverse.json"))
subset = {k: v for k, v in mapping.items() if k in uids}
json.dump(subset, open("subset.json","w"))