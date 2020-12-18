#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/17 23:24
# @Author : gsg
# @Site : 
# @File : predict.py
# @Software: PyCharm
import gc
import os
import pandas as pd

import torch
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
feature_path = './feature/'
model_path = './model/'
data_path = './data/data_format1'
prediction_path = './prediction/'

model = torch.load(os.path.join(model_path, 'checkpoint'))
matrix = pd.read_csv(os.path.join(feature_path, 'features_test.csv'))

# train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix
print(test_data.head())
# train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']
del matrix
gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDataset(Dataset):
    def __init__(self):
        x = np.array(test_data)
        # print(x.shape)
        self.x_data = torch.from_numpy(x)
        self.len = self.x_data.shape[0]
        # print(self.len)

    def __getitem__(self, item):
        return self.x_data[item]

    def __len__(self):
        return self.len


test_set = DDataset()
test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)
preds = []
model.eval()

with torch.no_grad():
    for feature in test_loader:
        feature = feature.float().to(device)
        pred, _ = model(feature, 0)
        preds += pred.cpu().data

mmp = []
for i in preds:
    fuck = []
    for ii in i:
        fuck.append(ii.item())
    mmp.append(fuck[1])

print(mmp)

pred = pd.read_csv(os.path.join(data_path, 'test_format1.csv'))
pred['prob'] = mmp

pred.to_csv(os.path.join(prediction_path, 'pred_nn.csv'), index=False)
