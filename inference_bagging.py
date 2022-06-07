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
out_csv = sys.argv[2]

device = utils.GetDevice()

train_df = pd.read_csv(input_csv).dropna()
train_ids = set(train_df["id"])

def ProcessModel(ind, outd):
    global train_df, train_ids

    print("Processing model", ind)

    model_name = ("model_%d.pth" % ind)
    used_csv = ("train_%d.csv" % ind)

    used_df = pd.read_csv(used_csv).dropna()
    used_ids = set(used_df["index"])
    for id in used_ids:
        outd[id].append(None)
    ids_to_use = train_ids - used_ids
    df_to_use = train_df.loc[train_df.index[list(ids_to_use)]]
    print("df to use: ", len(df_to_use))

    inference = utils.Inference(model_name, device, df_to_use)

    for index, (id, _) in tqdm(enumerate(df_to_use.iterrows())):
        cl = inference.ClassifyOne(index)
        outd[id].append(cl)

outd = collections.defaultdict(list)
for ii in range(1,conf.BAGGING_COUNT+1):
    ProcessModel(ii, outd)

with open(out_csv, "w") as fout:
    csvw = csv.writer(fout)
    for id, l in outd.items():
        csvw.writerow([id]+l)
