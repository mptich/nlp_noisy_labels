import sys
import pandas as pd

import conf
import utils
import network


assert len(sys.argv) == 3
train_file = sys.argv[1]
out_pth = sys.argv[2]

test_file = "test.csv"

train_df = pd.read_csv(train_file).dropna()
test_df = pd.read_csv(test_file).dropna()

print("Confusions: ", len(train_df[train_df["label"] != train_df["actual_label"]]))

print("Training Data")
print(train_df.head())
print("\n\nTest Data")
print(test_df.head())

###################################

import numpy as np
import torch
from typing import List


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset: pd.DataFrame, encodings, label_column: str = "label"
    ):
        self.dataset = dataset
        self.encodings = encodings
        self.labels = np.array(
            [utils.LabelToClass(label) for label in self.dataset[label_column]]
        )

    def MaxLength(self):
        return self.encodings.Length()

    def __getitem__(self, idx):
        x, attention_mask = utils.GetInputsAttention(self.encodings(), idx)
        y = self.labels[idx]
        return self.dataset["id"][idx], x, attention_mask, y, self.encodings.TextList()[idx]

    def __len__(self):
        return len(self.dataset)

train_enc = utils.PretrainedEncodings(train_df, max_length=conf.MAX_TOKEN_LEN)
test_enc = utils.PretrainedEncodings(test_df, max_length=conf.MAX_TOKEN_LEN)

train_dataset = TextDataset(train_df, train_enc, label_column="label")
test_dataset = TextDataset(
    test_df, 
    test_enc
)
print("MAX LENGTH: ", train_dataset.MaxLength())
assert conf.MAX_TOKEN_LEN == train_dataset.MaxLength()

print("train len", len(train_dataset), "validation len", len(test_dataset))

#####################################

import torch
import torch.nn.functional as F
import numpy as np
import random
import torch
import torch.nn.functional as F

print("Before torchmetrics")
import torchmetrics
from tqdm import tqdm

print("Start main program")

BATCH_SIZE = 32
NUM_EPOCHS = 4

device = utils.GetDevice()

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
)

print("Created data loaders")

model = network.TextClassificationModel(num_labels=conf.CLASS_COUNT)
print("Created model")
model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

best_valid_acc = 0.

for epoch in range(NUM_EPOCHS):
    print("Training epoch", epoch)

    model.train()
    running_loss = 0.0
    train_acc = torchmetrics.Accuracy()

    for index, data in enumerate(tqdm(train_dataloader)):
        x_idxs, x, attention_mask, y, txt = data
        x = x.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        log_probs = model(x, attention_mask)
        loss = F.nll_loss(log_probs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        log_probs_argmax = torch.argmax(log_probs.to("cpu"), 1)
        train_acc.update(log_probs_argmax, y.to("cpu"))

    model.eval()
    with torch.no_grad():
        valid_loss_accum = 0.0
        val_acc = torchmetrics.Accuracy()

        for data in tqdm(val_dataloader):
            x_idxs, x, attention_mask, y, txt = data

            x = x.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)

            log_probs = model(x, attention_mask)
            loss = F.nll_loss(log_probs, y)
            valid_loss_accum += loss.item()

            log_probs_argmax = torch.argmax(log_probs.to("cpu"), 1)
            val_acc.update(log_probs_argmax, y.to("cpu"))

        print("[epoch %d] Validation loss: %.3f" % (epoch, valid_loss_accum))
        validation_acc = val_acc.compute().item()
        training_acc = train_acc.compute().item()
        print("Current train acc: ", training_acc, " valid acc: ", validation_acc)
        if validation_acc > best_valid_acc:
            print("Saving...")
            torch.save(model.state_dict(), out_pth)
            acc_file_name = ("%s.accur" % out_pth)
            with open(acc_file_name, "w") as fout:
                fout.write("%.04f" % validation_acc)
            best_valid_acc = validation_acc
       
print("Finished Training")
