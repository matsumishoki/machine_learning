# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:19:13 2015

@author: matsumi
"""

import load_mnist
import numpy as np
import matplotlib.pyplot as plt


def softmax(s):
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s)


def onehot(k, num_classes=10):
    t_onehot = np.zeros(num_classes)
    t_onehot[k] = 1
    return t_onehot

# main文
if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    num_train, D = x_train.shape
    num_test = len(x_test)

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape
    print "x_test.shape:", x_test.shape
    print "t_test.shape:", t_test.shape

    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    # データ・セットの読み込み
    X_raw = x_train / 16.0
    num_examples = len(X_raw)
    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    x = X_raw[0]
    X_train = np.hstack((X_raw, np.ones((num_examples, 1))))
    dim_features = X_train.shape[-1]  # xの次元

    # learning_rateを定義する(learning_rate = 0.5で良いか判断し，収束しなければ値を変える．)
    learning_rate = 0.5

    # 収束するまで繰り返す
    max_iteration = 10000

    # dim_features次元の重みをnum_classesクラス分用意する
    w = np.random.randn(num_classes, dim_features)

    # 確率的勾配降下法
    error_history = []
    correct_percent_history = []
    for epoch in range(max_iteration):
        for x_i, t_i in zip(X_train, t_train):
            y_i = softmax(np.inner(w, x_i))
            T = onehot(t_i)
            w_new = w - learning_rate * np.expand_dims(y_i - T, 1) * x_i
            w = w_new

    # 負の対数尤度関数の値を表示する
        errors = []
        for x_i, t_i in zip(X_train, t_train):
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
        y = softmax(np.inner(X_train, w))
        predict_class = np.argmax(y, axis=1)
        num_correct = np.sum(t_train == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        print "epoch times:", epoch
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
    print "predict_class:", predict_class
    print "t:", t_train

    # wの可視化
