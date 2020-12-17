#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/17 13:25
# @Author : gsg
# @Site : 
# @File : another_try.py
# @Software: PyCharm
import os
import os
import numpy as np
import pandas as pd

data_format1_path = './data/data_format1/'
data_format2_path = './data/data_format2/'

user_log = pd.read_csv(os.path.join(data_format1_path, 'user_log_format1.csv'), dtype={'time_stamp': 'str'})
user_info = pd.read_csv(os.path.join(data_format1_path, 'user_info_format1.csv'))
train_data1 = pd.read_csv(os.path.join(data_format1_path, 'train_format1.csv'))
submission = pd.read_csv(os.path.join(data_format1_path, 'test_format1.csv'))
train_data = pd.read_csv(os.path.join(data_format2_path, 'train_format2.csv'))

print(user_log.head())
print(user_info.head())
print(train_data1.head())
print(submission.head())
print(train_data.head())