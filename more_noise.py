import sys
import collections
import pandas as pd
import random

in_csv = sys.argv[1]
out_csv = sys.argv[2]
prob = float(sys.argv[3])

train_data = in_csv
test_data = "test.csv"

train_df = pd.read_csv(train_data).dropna()#.sample(frac=0.5, replace=False)
test_df = pd.read_csv(test_data).dropna()

conf_df = train_df[train_df["label"] != train_df["actual_label"]]
print("Confusions: ", len(conf_df))

d = collections.defaultdict(list)
pairs = zip(conf_df["actual_label"], conf_df["label"])
for al, l in pairs:
 d[al].append(l)

same_df = train_df[train_df["label"] == train_df["actual_label"]]
for al, llist in d.items():
 al_df = same_df[same_df["actual_label"] == al]
 index_set = set(al_df["id"])
 for l in llist:
  index = random.choices([*index_set], k=1)[0]
  if random.random() <= prob:
   train_df.loc[index:index, "label"] = l
   index_set.remove(index)

train_df.to_csv(out_csv, index_label="id", index=False)
   
