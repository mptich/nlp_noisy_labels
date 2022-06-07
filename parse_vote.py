import sys
import csv
import pandas as pd
import collections

import conf
import utils

def FixTrainDataset(df, index, label):
    df.drop(index, axis=0, inplace=True)
    #df.loc[index, "label"] = label

train_file = sys.argv[1]
vote_csv = sys.argv[2]
new_train_file = sys.argv[3]

strong_votes = {}

with open(vote_csv, "r") as fin:
    csvr = csv.reader(fin)
    for l in csvr:
        assert len(l) == conf.BAGGING_COUNT + 1
        d = collections.Counter()
        index = int(l[0])
        # Method total() for Counter is available only starting from 3.10
        total_count = 0
        for cl in l[1:]:
            if cl:
                d[int(cl)] += 1
                total_count += 1
        if d:
            cl, count = d.most_common(1)[0]
            if count >= conf.BAGGING_VOTE_THRESH[total_count]:
                # Strong vote
                strong_votes[index] = cl

print("Strong votes: ", len(strong_votes))

df = pd.read_csv(train_file).dropna()

good_fix = 0
good_bad_fix = 0
bad_fix = 0

fix_dict = {}
for index, cl in strong_votes.items():
    if utils.LabelToClass(df.loc[index, "label"]) != cl:
        label = utils.ClassToLabel(cl)
        if label == df.loc[index, "actual_label"]:
            good_fix += 1
        elif df.loc[index, "actual_label"] != df.loc[index, "label"]:
            good_bad_fix += 1
        else:
            bad_fix += 1
        FixTrainDataset(df, index, label)

print("Fixes: good %d, good_bad %d, bad %d" % (good_fix, good_bad_fix, bad_fix))          

df = df.drop("id", axis=1)
df.reset_index(inplace=True, drop=False)
df = df.drop("index", axis=1)
df.to_csv(new_train_file, index_label="id")
