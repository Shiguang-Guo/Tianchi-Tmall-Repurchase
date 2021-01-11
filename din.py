#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/18 18:15
# @Author : gsg
# @Site : 
# @File : din.py
# @Software: PyCharm

import gc

import os

import numpy as np
import pandas as pd
import ast
import tensorflow as tf

data_format1_path = './data/data_format1/'
data_format2_path = './data/data_format2/'
feature_path = './feature/'


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


# user_log = pd.read_csv(os.path.join(data_format1_path, 'user_log_format1.csv'), dtype={'time_stamp': 'str'})
# user_info = pd.read_csv(os.path.join(data_format1_path, 'user_info_format1.csv'))
# train_data1 = pd.read_csv(os.path.join(data_format1_path, 'train_format1.csv'))
submission = pd.read_csv(os.path.join(data_format1_path, 'test_format1.csv'))
# train_data = pd.read_csv(os.path.join(data_format2_path, 'train_format2.csv'))
# train_data1['origin'] = 'train'
submission['origin'] = 'test'
# matrix = pd.concat([train_data1, submission], ignore_index=True, sort=False)
#
# # 使用merchant_id（原列名seller_id）
# user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
#
# # 格式化
# user_log['user_id'] = user_log['user_id'].astype('int32')
# user_log['merchant_id'] = user_log['merchant_id'].astype('int32')
# user_log['item_id'] = user_log['item_id'].astype('int32')
# user_log['cat_id'] = user_log['cat_id'].astype('int32')
# user_log['brand_id'].fillna(0, inplace=True)
# user_log['brand_id'] = user_log['brand_id'].astype('int32')
# user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%H%M')
#
# lbe_merchant_id = LabelEncoder()
# lbe_merchant_id.fit(np.r_[0, user_log['merchant_id'].values])
# user_log['merchant_id'] = lbe_merchant_id.transform(user_log['merchant_id'])
# matrix['merchant_id'] = lbe_merchant_id.transform(matrix['merchant_id'])
#
# lbe_user_id = LabelEncoder()
# user_log['user_id'] = lbe_user_id.fit_transform(user_log['user_id'])
# user_info['user_id'] = lbe_user_id.transform(user_info['user_id'])
# matrix['user_id'] = lbe_user_id.transform(matrix['user_id'])
#
# lbe_item_id = LabelEncoder()
# user_log['item_id'] = lbe_item_id.fit_transform(user_log['item_id'])
# lbe_cat_id = LabelEncoder()
# user_log['cat_id'] = lbe_cat_id.fit_transform(user_log['cat_id'])
# lbe_brand_id = LabelEncoder()
# user_log['brand_id'] = lbe_brand_id.fit_transform(user_log['brand_id'])
#
# user_log['merchant_id'].max(), user_log['user_id'].max()
# matrix = matrix.merge(user_info, on='user_id', how='left')
#
# matrix['age_range'].fillna(0, inplace=True)
# # 0:female, 1:male, 2:unknown
# matrix['gender'].fillna(2, inplace=True)
# matrix['age_range'] = matrix['age_range'].astype('int8')
# matrix['gender'] = matrix['gender'].astype('int8')
# matrix['label'] = matrix['label'].astype('str')
# matrix['user_id'] = matrix['user_id'].astype('int32')
# matrix['merchant_id'] = matrix['merchant_id'].astype('int32')
# del user_info, train_data1
# gc.collect()
# print(matrix)
#
# groups = user_log.groupby(['user_id'])
#
# # 用户交互行为数量 u1
# temp = groups.size().reset_index().rename(columns={0: 'u1'})
# matrix = matrix.merge(temp, on='user_id', how='left')
#
# # 使用agg 基于列的聚合操作，统计唯一值的个数 item_id, cat_id, merchant_id, brand_id
# # temp = groups['item_id', 'cat_id', 'merchant_id', 'brand_id'].nunique().reset_index().rename(columns={'item_id':'u2', 'cat_id':'u3', 'merchant_id':'u4', 'brand_id':'u5'})
#
# temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
# matrix = matrix.merge(temp, on='user_id', how='left')
# temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
# matrix = matrix.merge(temp, on='user_id', how='left')
# temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
# matrix = matrix.merge(temp, on='user_id', how='left')
# temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
# matrix = matrix.merge(temp, on='user_id', how='left')
#
# # 时间间隔特征 u6 按照小时
# temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
# temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds / 3600
# matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')
# # 统计action_type为0，1，2，3的个数（原始操作，没有补0）
# temp = groups['action_type'].value_counts().unstack().reset_index().rename(
#     columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
# matrix = matrix.merge(temp, on='user_id', how='left')
# print(matrix)
#
# groups = user_log.groupby(['merchant_id'])
#
# # 商家被交互行为数量 m1
# temp = groups.size().reset_index().rename(columns={0: 'm1'})
# matrix = matrix.merge(temp, on='merchant_id', how='left')
#
# # 统计商家被交互的user_id, item_id, cat_id, brand_id 唯一值
# temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(
#     columns={'user_id': 'm2', 'item_id': 'm3', 'cat_id': 'm4', 'brand_id': 'm5'})
# matrix = matrix.merge(temp, on='merchant_id', how='left')
#
# # 统计商家被交互的action_type 唯一值
# temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0: 'm6', 1: 'm7', 2: 'm8', 3: 'm9'})
# matrix = matrix.merge(temp, on='merchant_id', how='left')
#
# # 按照merchant_id 统计随机负采样的个数
# temp = train_data[train_data['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(columns={0: 'm10'})
# matrix = matrix.merge(temp, on='merchant_id', how='left')
#
# # 按照user_id, merchant_id分组
# groups = user_log.groupby(['user_id', 'merchant_id'])
# temp = groups.size().reset_index().rename(columns={0: 'um1'})  # 统计行为个数
# matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
# temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(
#     columns={'item_id': 'um2', 'cat_id': 'um3', 'brand_id': 'um4'})  # 统计item_id, cat_id, brand_id唯一个数
# matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
# temp = groups['action_type'].value_counts().unstack().reset_index().rename(
#     columns={0: 'um5', 1: 'um6', 2: 'um7', 3: 'um8'})  # 统计不同action_type唯一个数
# matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
# temp = groups['time_stamp'].agg([('first', 'min'), ('last', 'max')]).reset_index()
# temp['um9'] = (temp['last'] - temp['first']).dt.seconds / 3600
# temp.drop(['first', 'last'], axis=1, inplace=True)
# print(temp)
# print('-' * 100)
# matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')  # 统计时间间隔
# print(matrix)
#
# # 用户购买点击比
# matrix['r1'] = matrix['u9'] / matrix['u7']
# # 商家购买点击比
# matrix['r2'] = matrix['m8'] / matrix['m6']
# # 不同用户不同商家购买点击比
# matrix['r3'] = matrix['um7'] / matrix['um5']
# matrix.fillna(0, inplace=True)
# # 修改age_range字段名称为 age_0, age_1, age_2... age_8
# temp = pd.get_dummies(matrix['age_range'], prefix='age')
# matrix = pd.concat([matrix, temp], axis=1)
# temp = pd.get_dummies(matrix['gender'], prefix='g')
# matrix = pd.concat([matrix, temp], axis=1)
# matrix.drop(['age_range', 'gender'], axis=1, inplace=True)
#
#
#
# lbe_action_type = {0: 1, 1: 2, 2: 3, 3: 4}
# user_log['action_type'] = user_log['action_type'].map(lbe_action_type)
# # 用户行为sequence
# # 把user_log里同user的这些数据合并成一个list
# temp = pd.DataFrame(user_log.groupby('user_id')['merchant_id', 'action_type'].agg(lambda x: list(x)))
# # 列名称改成hist_merchant_id 和 hist_action_type
# temp.columns = ['hist_merchant_id', 'hist_action_type']
# matrix = matrix.merge(temp, on=['user_id'], how='left')  # 统计时间间隔
#
# # 截取，不缺到定长M个
M = 500
# for feature in ['hist_merchant_id', 'hist_action_type']:
#     matrix[feature] = matrix[feature].map(lambda x: np.array(x + [0] * (M - len(x)))[:M])
#
# matrix.to_csv("matrix_din.csv")
# print("########################################")

matrix = pd.read_csv('matrix_din.csv', header=0, index_col=0)
matrix['hist_merchant_id'] = matrix['hist_merchant_id'].str.replace(r'\[\s+', '[').str.replace(r'\s+', ',').map(
    ast.literal_eval)
matrix['hist_action_type'] = matrix['hist_action_type'].str.replace(r'\[\s+', '[').str.replace(r'\s+', ',').map(
    ast.literal_eval)
print("########################################")

# 分割训练数据和测试数据
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models import DIN

train_X['action_type'] = 3

# print(train_X.drop_duplicates(['user_id']).shape)
# print(len(train_X.columns))
# print(train_X.columns)
# train_X = train_X.drop(labels=['prob', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6',
#                                'u7', 'u8', 'u9', 'u10', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8',
#                                'm9', 'm10', 'um1', 'um2', 'um3', 'um4', 'um5', 'um6', 'um7', 'um8',
#                                'um9', 'r1', 'r2', 'r3'], axis=1)
# print(train_X.columns)
feature_columns = []
for column in train_X.columns:
    if column != 'hist_merchant_id' and column != 'hist_action_type':
        num = train_X[column].nunique()
        if num > 10000:
            dim = 10
        else:
            if num > 1000:
                dim = 8
            else:
                dim = 4

        if column == 'user_id':
            feature_columns += [SparseFeat(column, 212062 + 1, embedding_dim=dim)]
        elif column == 'merchant_id':
            feature_columns += [SparseFeat(column, 1993 + 1, embedding_dim=dim)]
        elif column == 'action_type':
            feature_columns += [SparseFeat(column, 4 + 1, embedding_dim=dim)]
        else:
            feature_columns += [DenseFeat(column, 1)]

# maxlen为历史信息的长度，vocabulary_size为onehot的长度
feature_columns += [
    VarLenSparseFeat(sparsefeat=SparseFeat('hist_merchant_id', vocabulary_size=1993, embedding_dim=8,
                                           embedding_name='merchant_id'), maxlen=M),
    VarLenSparseFeat(sparsefeat=SparseFeat('hist_action_type', vocabulary_size=4, embedding_dim=4,
                                           embedding_name='action_type'), maxlen=M)]
# feature_columns += [VarLenSparseFeat('hist_merchant_id', maxlen=M, vocabulary_size=4994 + 1, embedding_dim=8,
#                                      embedding_name='merchant_id'),
#                     VarLenSparseFeat('hist_action_type', maxlen=M, vocabulary_size=4 + 1, embedding_dim=4,
#                                      embedding_name='action_type')]
hist_features = ['merchant_id', 'action_type']
print(len(feature_columns))

# 使用DIN模型
model = DIN(feature_columns, hist_features)
# 使用Adam优化器，二分类的交叉熵
# model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"])

# 组装train_model_input，得到feature names，将train_X转换为字典格式
feature_names = list(train_X.columns)
train_model_input = {name: train_X[name].values for name in get_feature_names(feature_columns)}
print(train_model_input['hist_merchant_id'][0].dtype)  # print(train_model_input)

print("########################################")

# histroy输入必须是二维数组
from tqdm import tqdm

for fea in ['hist_merchant_id', 'hist_action_type']:
    l = []
    for i in tqdm(train_model_input[fea]):
        l.append(i)
    train_model_input[fea] = np.array(l)

for key in train_model_input:
    # print(type(key))
    print(train_model_input[key].dtype)
# print(type(train_y[0].values))
history = model.fit(train_model_input, train_y.values, verbose=True, epochs=10, validation_split=0.2, batch_size=512)

# 转换test__model_input
test_data['action_type'] = 3
test_model_input = {name: test_data[name].values for name in feature_names}
from tqdm import tqdm

for fea in ['hist_merchant_id', 'hist_action_type']:
    l = []
    for i in tqdm(test_model_input[fea]):
        l.append(i)
    test_model_input[fea] = np.array(l)

# 得到预测结果
prob = model.predict(test_model_input)
submission['prob'] = prob
submission.drop(['origin'], axis=1, inplace=True)
submission.to_csv('prediction.csv', index=False)
