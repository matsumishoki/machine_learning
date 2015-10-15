# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:05:52 2015

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
