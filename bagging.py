import sys
import collections
import pandas as pd

input_csv = sys.argv[1]
train_csv = sys.argv[2]

train_data = input_csv
test_data = "test.csv"

train_df = pd.read_csv(train_data).dropna()#.sample(frac=0.5, replace=False)
test_df = pd.read_csv(test_data).dropna()

count = len(train_df)
train_df = train_df.sample(n=count//3, replace=True)
train_df = train_df.drop("id", axis=1)
train_df.reset_index(inplace=True, drop=False)
train_df.to_csv(train_csv, index_label="id")
