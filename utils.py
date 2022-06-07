import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer

import conf
import network

class PretrainedEncodings:

    def __init__(self, df: pd.DataFrame,
        pretrained_model: str = "distilbert-base-uncased", max_length: int = 0):

        self.text_list = df["text"].tolist()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        if not max_length:
            self.encodings = tokenizer(
                self.text_list, truncation=False, padding=True
            )
        else:
            self.encodings = tokenizer(
                self.text_list, truncation=False, padding="max_length", max_length=max_length
            )
    
    def __call__(self):
        return self.encodings
 
    def Length(self):
        return len(self.encodings["input_ids"][0])

    def TextList(self):
        return self.text_list

def GetInputsAttention(encodings, idx):
    inputs = torch.tensor(encodings["input_ids"][idx])
    attention_mask = torch.tensor(encodings["attention_mask"][idx])
    return inputs, attention_mask

def GetDevice():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def LabelToClass(label):
    ind = conf.CLASSES.index(label)
    assert ind >= 0
    return ind

def ClassToLabel(cl):
    return conf.CLASSES[cl]

class Inference:
    def __init__(self, model_name, device, df):
        self.device = device
        self.df = df
        self.model = network.TextClassificationModel(num_labels=conf.CLASS_COUNT)
        self.model.load_state_dict(torch.load(model_name))
        self.model.to(self.device)
        self.encodings = PretrainedEncodings(df, max_length=conf.MAX_TOKEN_LEN)

    def ClassifyOne(self, index):
        inputs, attention_mask = GetInputsAttention(self.encodings(), index)
        inputs = inputs.to(self.device).unsqueeze_(0)
        attention_mask = attention_mask.to(self.device).unsqueeze_(0)
        log_probs = self.model(inputs, attention_mask).squeeze_(0)
        cl = torch.argmax(log_probs).to("cpu").item()
        return cl

    def ClassifyAll(self):
        d = {}
        for index, (id, _) in tqdm(enumerate(self.df.iterrows())):
            cl = self.ClassifyOne(index)
            d[id] = cl
        return d
