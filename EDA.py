#!/usr/bin/env python

from dataset import *
import numpy as np
import sys
import torch

dataset = SatelliteImages("..")
# nan_count = 0
# for _, i in dataset.train_data:
#     if i is np.nan:
#         nan_count += 1
# print(str(nan_count)+"/"+str(len(dataset)))  # 75000/131030

def check_mean_std():
    dataset = SatelliteImages("..")
    mean = np.zeros(3, dtype=np.int64).astype(np.float64)
    std = np.zeros(3, dtype=np.int64).astype(np.float64)
    num = 0

    for i, j in dataset:
        k = (torch.sum(i, dim=[1, 2]) / i.numel()).numpy()
        std += k*k
        mean += k
        num += 1
        sys.stdout.write("\r" + str(num) + " " + str(list(mean / num)) + "\t" + str(list((std/num - mean/num * mean/num))))
        # 8369 [0.20478993657609046, 0.28879167546050427, 0.31729726713007855]	[0.033844949144484056, 0.027076033017128187, 0.019965087489821884]
        sys.stdout.flush()
