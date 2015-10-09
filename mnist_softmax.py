# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:19:13 2015

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time


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

    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    v = train_test_split(x_train, t_train, test_size=10000, random_state=42)
    x_train, x_valid, t_train, t_valid = v

    num_train, D = x_train.shape
    num_valid = len(x_valid)
    num_test = len(x_test)

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape
    print "x_valid.shape:", x_valid.shape
    print "t_valid.shape:", t_valid.shape
    print "x_test.shape:", x_test.shape
    print "t_test.shape:", t_test.shape

    # 訓練データ・セットの読み込み
    X_raw = x_train
    num_examples = len(X_raw)
    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    x = X_raw[0]
    X_train = np.hstack((X_raw, np.ones((num_examples, 1))))
    dim_features = X_train.shape[-1]  # xの次元

    # テストデータ・セットの読み込み
    num_test_examples = len(x_test)
    test_classes = np.unique(t_test)
    num_test_classes = len(test_classes)
    X_test = np.hstack((x_test, np.ones((num_test_examples, 1))))
    dim_features = X_test.shape[-1]  # xの次元

    # learning_rateを定義する(learning_rate = 0.5で良いか判断し，収束しなければ値を変える．)
    learning_rate = 0.0019

    # 収束するまで繰り返す
    max_iteration = 100000

    # dim_features次元の重みをnum_classesクラス分用意する

    w_scale = 0.001
    w = w_scale * np.random.randn(num_classes, dim_features)

    # 確率的勾配降下法
    error_history = []
    correct_percent_history = []
    for epoch in range(max_iteration):
        print "epoch:", epoch

        time_start = time.time()
        perm = np.random.permutation(num_train)
        for i in perm:
            x_i = X_train[i]
            t_i = t_train[i]
            y_i = softmax(np.inner(w, x_i))
            T = onehot(t_i)
            w_new = w - learning_rate * np.expand_dims(y_i - T, 1) * x_i
            w = w_new

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 負の対数尤度関数の値を表示する
        errors = []
        for x_i, t_i in zip(X_train, t_train):
            y = softmax(np.inner(x_i, w))
            T = onehot(t_i)
            error = np.sum(-(T*(np.log(y))))
            errors.append(error)
            assert not np.any(np.isnan(error))
            assert not np.any(np.isinf(error))

        total_error = np.mean(errors)
        print "error:", total_error
        error_history.append(total_error)

        # 正解クラスと予測クラスとの比較
        y = softmax(np.inner(X_train, w))
        predict_class = np.argmax(y, axis=1)
        num_correct = np.sum(t_train == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        print "|w|:", np.linalg.norm(w)
        correct_percent_history.append(correct_percent)

        # 学習曲線をプロットする
        plt.plot(error_history)
        plt.title("error")
        plt.legend(["error"])
        plt.grid()
        plt.show()
        plt.plot(correct_percent_history)
        plt.title("correct_percent")
        plt.legend(["correct_percent"], loc="lower right")
        plt.grid()
        plt.show()

        if epoch == 1:
            break

    # 訓練データセットとテストデータセットとの比較
    y_test = softmax(np.inner(X_test, w))
    predict_class_test = np.argmax(y_test, axis=1)
    num_correct_test = np.sum(t_test == predict_class_test)
    correct_softmax_percent = num_correct_test / float(num_test_examples) * 100
    print "correct_softmax_percent:", correct_softmax_percent
    print "finish epoch:", epoch
    print "|w|:", np.linalg.norm(w)

    # 予測クラスと真のクラスを表示する
    print "predict_class:", predict_class
    print "t:", t_train

    # wの可視化
