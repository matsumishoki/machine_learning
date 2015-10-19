# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:52:01 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


X_train = np.arange(10*3).reshape((10, 3))
num_train = len(X_train)
batch_size = 3
num_batches = num_train / batch_size
max_iteration = 2

for epoch in range(max_iteration):
    print "epoch:", epoch
    perm = np.random.permutation(num_train)
    for batch_indexes in np.array_split(perm, num_batches):
        X_batch = X_train[batch_indexes]

        print "batch_indexes:", batch_indexes
        print "X_batch:"
        print X_batch
        print
