#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/17 16:23
# @Author : gsg
# @Site : 
# @File : model.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torch import nn


class mlp(nn.Module):
    def __init__(self, labels):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(labels, 32)
        # self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32, 2)
        self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        self.focalloss = FocalLoss(class_num=2, alpha=0.25,)

    def forward(self, x, label):
        # print(x.shape)
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.dropout1(x)
        # x = F.tanh(self.linear2(x))
        # x = self.dropout2(x)
        x = F.softmax(self.linear3(x))
        loss = self.focalloss(x, label)
        return x, loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.ones(class_num, 1) * alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        if self.training:
            N = inputs.size(0)
            C = inputs.size(1)

            class_mask = inputs.data.new(N, C).fill_(0)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (inputs * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss
