# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:52:01 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def onehot(k, num_classes=10):
    num_examples = len(k)
    t_onehot = np.zeros((num_examples, num_classes))
    t_onehot[np.arange(num_examples), k] = 1
    return t_onehot

X_train = np.arange(10*3).reshape((10, 3))
