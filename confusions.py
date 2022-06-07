import sys
import collections
import pandas as pd
import numpy as np

train_data = sys.argv[1]
test_data = "test.csv"

train_df = pd.read_csv(train_data).dropna()#.sample(frac=0.5, replace=False)
test_df = pd.read_csv(test_data).dropna()

uniqs, counts = np.unique(train_df["actual_label"], return_counts=True)
total = np.sum(counts)
act_totals = {}
act_fractions = {}
for ii, act_label in enumerate(uniqs):
    act_totals[act_label] = counts[ii]
    fraction = counts[ii] / total
    act_fractions[act_label] = fraction
    print("%s: %0.3f" % (act_label, fraction))

conf_df = train_df[train_df["label"] != train_df["actual_label"]]
print("Confusions: %d, fraction of confusions %0.3f" % (len(conf_df), len(conf_df) / total))

pairs = zip(conf_df["actual_label"], conf_df["label"])

conf_by_act = collections.Counter()
conf_by_tup = collections.defaultdict(collections.Counter)
for p in pairs:
    conf_by_act[p[0]] += 1
    conf_by_tup[p[0]][p] += 1
for actl, count in conf_by_act.most_common():
    print("Fraction of %s confused: %0.3f" % (actl, count / act_totals[actl]))
    for lab in uniqs:
        if lab != actl:
            print("Adjusted freq of confusion with %s: %0.3f" % (lab, conf_by_tup[actl][(actl,lab)] / count * (1 - act_fractions[actl])))

