import sys
import collections
import pandas as pd
import csv
import torch
from tqdm import tqdm

import network
import conf
import utils

input_csv = sys.argv[1]
model_file = sys.argv[2]
new_train_file = sys.argv[3]

device = utils.GetDevice()

df = pd.read_csv(input_csv).dropna()
inference = utils.Inference(model_file, device, df)
dout = inference.ClassifyAll()

good_fix = 0
bad_to_bad = 0
bad_fix = 0
for id, cl in dout.items():
    inf_label = utils.ClassToLabel(cl)
    act_label = df.loc[id, "actual_label"]
    label = df.loc[id, "label"]
    df.loc[id, "label"] = inf_label
    if inf_label == act_label:
        if act_label != label:
            good_fix += 1
    else:
        if act_label != label:
            bad_to_bad += 1
        else:
            bad_fix += 1
           
print("Good fix %d, bad to bad %d, bad fix %d" % (good_fix, bad_to_bad, bad_fix))

df.to_csv(new_train_file, index_label="id", index=False)
