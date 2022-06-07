import sys
import pandas as pd

fname = sys.argv[1]

df = pd.read_csv(fname).dropna()
for column in df.columns:
    if column not in ("id", "index", "text", "label", "actual_label"):
        df.drop(column, axis=1, inplace=True)

df.to_csv(fname, index_label="id", index=False)

