# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 20:22:25 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def sigmoid(s):
    y = 1 / (1 + np.exp(-s))
    return y


# main文
if __name__ == '__main__':
    digits = load_digits(2)
    images = digits.images
    plt.matshow(images[4], cmap=plt.cm.gray)
    plt.show()

    # データ・セットの読み込み
    X_raw = digits.data
    t = digits.target
    x = X_raw[0]
    X = np.hstack((X_raw, np.ones((360, 1))))

    # ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)
    rho = 0.1

    # wを定義する（65個の個数を持った配列をrandomで作成する）
    w = np.random.randn(65)

    # wとXの内積を計算する
    y = np.inner(w, X)

    # print y
    # print sigmoid(y)

    # 合成関数の対数をとる
    L = np.log(sigmoid(y))
    print L
