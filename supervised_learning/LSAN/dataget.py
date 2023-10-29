import re
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("../../bert-base-chinese")
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text=text.replace(" ","")
    text=text.replace("nbsp","")
    return text

class MyData(Dataset):
    def __init__(self, path="cognitive_distortion_train_LSAN.csv"):
        datas = pd.read_csv(path)
        self.texts = datas.pop('内容').apply(preprocess_text).values
        self.labels = datas.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sentence = str(self.texts[item])
        label = np.int64(self.labels[item])
        return sentence, label

    @staticmethod
    def call_fc(batch):
        xs = []
        ys = []
        for x, y in batch:
            xs.append(x)
            ys.append(y)
        input_ids = tokenizer.batch_encode_plus(xs, truncation=True, padding=True, return_tensors="np")['input_ids']
        labels = np.array(ys)
        input_ids = torch.IntTensor(input_ids)
        labels = torch.LongTensor(labels)
        return input_ids, labels

def create_data(path='cognitive_distortion_train_LSAN.csv', batch_size=12):
    data = MyData(path)
    data = DataLoader(data, shuffle=True, batch_size=batch_size, collate_fn=data.call_fc, drop_last=True)
    return data

