import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import os
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import tqdm
import logging
import argparse
import time
import logging
import wandb

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = metrics.roc_curve(y_true=label, y_score=pred, pos_label=positive_label)
    fnr = 1 - tpr

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


class FCN(nn.Module):
    def __init__(self, data_len, num_features=4, num_class=2):
        super(FCN, self).__init__()
        self.num_class = num_class

        self.c1 = nn.Conv1d(num_features, 128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(128)

        self.c2 = nn.Conv1d(128, 256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(256)

        self.c3 = nn.Conv1d(256, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(data_len-13, num_class)

        # self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.c1(x)        
        x = self.relu(self.bn1(x))

        x = self.c2(x)
        x = self.relu(self.bn2(x))

        x = self.c3(x)
        x = self.relu(self.bn3(x))
        x = x.transpose(1, 2)

        # x = nn.functional.adaptive_avg_pool2d(x, (1, self.num_class))
        # x = self.softmax(x.squeeze(1))

        x = torch.mean(x, 2)
        x = self.fc(x.reshape(x.size()[0], -1)) 


        return x