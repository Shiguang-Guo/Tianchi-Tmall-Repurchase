#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/17 16:23
# @Author : gsg
# @Site : 
# @File : model.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class mlp(nn.Module):
    def __init__(self, labels):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(labels, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.sigmoid(self.linear3(x))
        return x
