# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 17:07:32 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def softmax(s):
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s)


def onehot(k, num_classes=10):
    t_onehot = np.zeros(num_classes)
    t_onehot[k] = 1
    return t_onehot

# main文
if __name__ == '__main__':
    digits = load_digits()
    images = digits.images
    plt.matshow(images[4], cmap=plt.cm.gray)
    plt.show()

    # データ・セットの読み込み
    X_raw = digits.data / 16.0
    t = digits.target
    num_examples = len(X_raw)
    classes = np.unique(t)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    x = X_raw[0]
    X = np.hstack((X_raw, np.ones((num_examples, 1))))
    dim_features = X.shape[-1]  # xの次元
    # ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)
    rho = 0.5

    # 収束するまで繰り返す
    max_iteration = 1000

    # dim_features次元の重みをnum_classesクラス分用意する
    w = np.random.randn(num_classes, dim_features)

    # 確率的勾配降下法
    error_history = []
    correct_percent_history = []
    for epoch in range(max_iteration):
        for x_i, t_i in zip(X, t):
            y_i = softmax(np.inner(w, x_i))
            T = onehot(t_i)
            w_new = w - rho * np.expand_dims(y_i - T, 1) * x_i
            w = w_new
        # print w_new


        #負の対数尤度関数の値を表示する
        errors = []
        for x_i, t_i in zip(X, t):
            y = softmax(np.inner(x_i, w))
            T = onehot(t_i)
            error = np.sum(-(T*(np.log(y))))
            errors.append(error)
            assert not np.any(np.isnan(error))
            assert not np.any(np.isinf(error))

        total_error = sum(errors)
        print "error:", total_error
        error_history.append(total_error)

        # 正解クラスと予測クラスとの比較
        y = softmax(np.inner(X, w))
        predict_class = np.argmax(y, axis=1)
        num_correct = np.sum(t == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        correct_percent_history.append(correct_percent)

        # 学習曲線をプロットする
        plt.plot(error_history)
        plt.title("error")
        plt.legend(["error"])
        plt.show()
        plt.plot(correct_percent_history)
        plt.title("correct_percent")
        plt.legend(["correct_percent"], loc="lower right")
        plt.show()

        if correct_percent == 100.0:
            break

    # 予測クラスと真のクラスを表示する


    # wの可視化