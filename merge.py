#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/18 19:20
# @Author : gsg
# @Site : 
# @File : merge.py
# @Software: PyCharm

# 对数据按照格式进行压缩重新存储
import os
import pickle
import pandas as pd


def compressData(inputData):
    '''
    :parameters: inputData: pd.Dataframe
    :return: inputData: pd.Dataframe
    :Purpose:
    压缩csv中的数据，通过改变扫描每列的dtype，转换成适合的大小
    例如: int64, 检查最小值是否存在负数，是则声明signed，否则声明unsigned，并转换更小的int size
    对于object类型，则会转换成category类型，占用内存率小
    参考来自：https://www.jiqizhixin.com/articles/2018-03-07-3
    '''
    for eachType in set(inputData.dtypes.values):
        ##检查属于什么类型
        if 'int' in str(eachType):
            ## 对每列进行转换
            for i in inputData.select_dtypes(eachType).columns.values:
                if inputData[i].min() < 0:
                    inputData[i] = pd.to_numeric(inputData[i], downcast='signed')
                else:
                    inputData[i] = pd.to_numeric(inputData[i], downcast='unsigned')
        elif 'float' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values:
                inputData[i] = pd.to_numeric(inputData[i], downcast='float')
        elif 'object' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values:
                inputData[i] = trainData7[i].astype('category')
    return inputData


userInfo = pd.read_csv('d:/JulyCompetition/input/user_log_format1.csv')
print('Before compressed:\n', userInfo.info())
userInfo = compressData(userInfo)
print('After compressed:\n', userInfo.info())

userInfo.isnull().sum()


# brand_id使用所在seller_id对应的brand_id的众数填充
def get_Logs():
    '''
    :parameters: None: None
    :return: userLog: pd.Dataframe
    :Purpose:
    方便与其他函数调取原始的行为数据，同时已对缺失省进行调整
    使用pickle模块进行序列话，加快速度读写
    '''
    filePath = 'd:/JulyCompetition/features/Logs.pkl'
    if os.path.exists(filePath):
        userLog = pickle.load(open(filePath, 'rb'))
    else:
        userLog = pd.read_csv('d:/JulyCompetition/input/user_log_format1.csv', dtype=column_types)
        print('Is null? \n', userLog.isnull().sum())

        ## 对brand_id缺失值进行处理
        missingIndex = userLog[userLog.brand_id.isnull()].index
        ## 思路：找到所有商店所拥有brand_id的众数，并对所缺失的brand_id与其相对应的商店进行填充
        sellerMode = userLog.groupby(['seller_id']).apply(lambda x: x.brand_id.mode()[0]).reset_index()
        pickUP = userLog.loc[missingIndex]
        pickUP = pd.merge(pickUP, sellerMode, how='left', on=['seller_id'])[0].astype('float32')
        pickUP.index = missingIndex
        userLog.loc[missingIndex, 'brand_id'] = pickUP
        del pickUP, sellerMode, missingIndex
        print('--------------------')
        print('Is null? \n', userLog.isnull().sum())
        pickle.dump(userLog, open(filePath, 'wb'))
    return userLog


userLog = get_Logs()
