import sys
import pandas as pd

import conf
import utils


train_file = sys.argv[1]
test_file = sys.argv[2]

train_df = pd.read_csv(train_file).dropna()
test_df = pd.read_csv(test_file).dropna()

train_text_list = train_df["text"].tolist()
test_text_list = test_df["text"].tolist()
enc = utils.PretrainedEncodings(train_text_list+test_text_list)
print(enc.Length())
