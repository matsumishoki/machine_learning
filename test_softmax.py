# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:12:54 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def softmax(s):
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


x = np.array([1, 2, 3])
y = softmax(x)
y_expect = np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))
assert np.allclose(y, y_expect)

x = np.arange(5)
y = softmax(x)
assert y.shape == (5,)

x = np.arange(50).reshape(10, 5)
y = softmax(x)
assert y.shape == (10, 5)
