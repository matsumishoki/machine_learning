# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 20:22:25 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# main文
if __name__ == '__main__':
    digits = load_digits(2)
    images = digits.images
    plt.matshow(images[4], cmap=plt.cm.gray)
    plt.show()

    # シグモイド関数を定義する
    def sigmoid(s):
        y = 1 / (1 + np.exp(-s))
        return y

    # データ・セットの読み込み
    X_raw = digits.data
    t = digits.target
    x = X_raw[0]
    X = np.hstack((X_raw, np.ones((360, 1))))
