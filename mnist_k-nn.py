# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:52:14 2016

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    num_train = len(x_train)
    num_valid = len(x_valid)
    num_test = len(x_test)

    # wxの中に定数項であるバイアス項を入れ込む
    x_train = np.hstack((x_train, np.ones((num_train, 1))))  # (N×D)
    x_valid = np.hstack((x_valid, np.ones((num_valid, 1))))
    x_test = np.hstack((x_test, np.ones((num_test, 1))))

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元

    # 超パラメータの定義
