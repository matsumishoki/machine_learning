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
    X_raw = digits.data / 16.0
    t = digits.target
    num_examples = len(X_raw)
    x = X_raw[0]
    X = np.hstack((X_raw, np.ones((num_examples, 1))))

    # ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)
    rho = 1.4

    # wを定義する（65個の個数を持った配列をrandomで作成する）
    w = np.random.randn(65)

    # 収束するまで繰り返す
    max_iteration = 100

    # 確率的勾配降下法
    for epoch in range(max_iteration):
        for x_i, t_i in zip(X, t):
            y_i = sigmoid(np.inner(w, x_i))
            w_new = w - rho * (y_i - t_i) * x_i
            w = w_new

        # 負の対数尤度関数の値を表示する
        y = sigmoid(np.inner(w, X))
        error = np.sum(-(t*(np.log(y)) + (1 - t)*np.log(1 - y)))
        # print "error:", error

        # 正解クラスと予測クラスとの比較
        predict_class = y >= 0.5
        num_correct = np.sum(t == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        if correct_percent == 100.0:
            break

    # 予測クラスと真のクラスを表示する
    print "predict_class:", predict_class
    print "t:", t == 1

    # wの可視化
    w_t = w[0:64]
    print "w_t:", w_t
    plt.matshow(w_t.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
