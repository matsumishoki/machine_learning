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
    rho = 0.01

    # wを定義する（65個の個数を持った配列をrandomで作成する）
    w = np.random.randn(65)
#    w[-1] = 0

    # 確率的勾配降下法
    for epoch in range(100):
        for x_i, t_i in zip(X, t):
            y_i = sigmoid(np.inner(w, x_i))
            w_new = w - rho * (y_i - t_i) * x_i
            w = w_new

#        e_i = -(t_i*(np.log(y_i)) + (1 - t_i)*np.log(1 - y_i))
#        print "t_i:", t_i, ",y_i:", y_i, "np.inner(w, x_i):", np.inner(w, x_i)
#        print "    E_i:", e_i, ", |w|:", np.linalg.norm(w)
#        assert not (np.isnan(e_i) or np.isinf(e_i)), "e_i == {}".format(e_i)

        y = sigmoid(np.inner(w, X))
        error = np.sum(-(t*(np.log(y)) + (1 - t)*np.log(1 - y)))
        print "error:", error

        # 正解クラスと予測クラスとの比較
        predict_class = y >= 0.5
        num_correct = np.sum(t == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent

    # wの可視化
    w_t = w[0:64]
    plt.matshow(w_t.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
    print "w_t:", w_t
