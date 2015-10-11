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
    v = train_test_split(x_train, t_train, test_size=15000, random_state=100)
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
    num_examples = len(x_train)
    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    x = x_train[0]
    X_train = np.hstack((x_train, np.ones((num_examples, 1))))
    dim_features = X_train.shape[-1]  # xの次元

    # 評価のデータセットの読み込み
    valid_classes = np.unique(t_valid)  # 定義されたクラスラベル
    num_valid_classes = len(valid_classes)
    X_valid = np.hstack((x_valid, np.ones((num_valid, 1))))
    dim_features = X_valid.shape[-1]  # xの次元

    # テストデータ・セットの読み込み
    test_classes = np.unique(t_test)  # 定義されたクラスラベル
    num_test_classes = len(test_classes)
    X_test = np.hstack((x_test, np.ones((num_test, 1))))
    dim_features = X_test.shape[-1]  # xの次元

    # learning_rateを定義する(learning_rate = 0.5で良いか判断し，収束しなければ値を変える．)
    learning_rate = 0.00001

    # 収束するまで繰り返す
    max_iteration = 100000

    # dim_features次元の重みをnum_classesクラス分用意する

    w_scale = 0.001
    w = w_scale * np.random.randn(num_classes, dim_features)

    # 確率的勾配降下法
    error_history = []
    correct_percent_history = []
    error_valid_history = []
    correct_valid_percent_history = []

    w_best = 0
    correct_valid_percent_best = 0
    max_epoch = 100
    total_valid_error_best = 10

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

        # 訓練セットの負の対数尤度関数の値を表示する
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

        # 検証セットの負の対数尤度関数の値を表示する
        valid_errors = []
        for x_vi, t_vi in zip(X_valid, t_valid):
            y_v = softmax(np.inner(x_vi, w))
            T = onehot(t_vi)
            valid_error = np.sum(-(T*(np.log(y_v))))
            valid_errors.append(valid_error)
            assert not np.any(np.isnan(valid_error))
            assert not np.any(np.isinf(valid_error))

        total_valid_error = np.mean(valid_errors)
        print "valid_error:", total_valid_error
        error_valid_history.append(total_valid_error)

        # 学習中のモデルで訓練セットを評価して正解率を求める
        y = softmax(np.inner(X_train, w))
        predict_class = np.argmax(y, axis=1)
        num_correct = np.sum(t_train == predict_class)
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        correct_percent_history.append(correct_percent)

        # 学習中のモデルで検証セットを評価して正解率を求める
        y_valid = softmax(np.inner(X_valid, w))
        predict_class_valid = np.argmax(y_valid, axis=1)
        num_correct_valid = np.sum(t_valid == predict_class_valid)
        correct_valid_percent = num_correct_valid / float(num_valid) * 100
        print "correct_valid_percent:", correct_valid_percent
        print "|w|:", np.linalg.norm(w)
        correct_valid_percent_history.append(correct_valid_percent)

        # 学習曲線をプロットする
        plt.plot(error_history, label="error")
        plt.plot(error_valid_history, label="valid_error")
        plt.title("error")
        plt.legend(['error', 'valid_error'])
        # plt.legend(["error"])
        plt.grid()
        plt.show()

        plt.plot(correct_percent_history, label="correct_percent")
        plt.plot(correct_valid_percent_history, label="correct_valid_percent")
        plt.legend(loc="lower right")
        plt.title("correct_percent")
        # plt.legend(["correct_percent"], loc="lower right")
        # plt.legend(["correct_valid_percent"], loc="lower left")
        plt.grid()
        plt.show()

        # 検証データのerror率が良ければwの値を保存し，wの最善値を格納する
        if total_valid_error <= total_valid_error_best:
            w_best = w
            total_valid_error_best = total_valid_error
            print "valid_error_best:", total_valid_error_best
            if correct_valid_percent >= correct_valid_percent_best:
                correct_valid_percent_best = correct_valid_percent
                print "correct_valid_percent_best:", correct_valid_percent_best

        if epoch == max_epoch:
            break

    # 学習済みのモデルでテストセットを評価して正解率を求める
    y_test = softmax(np.inner(X_test, w_best))
    predict_class_test = np.argmax(y_test, axis=1)
    num_correct_test = np.sum(t_test == predict_class_test)
    correct_softmax_percent = num_correct_test / float(num_test) * 100
    print "correct_softmax_percent:", correct_softmax_percent
    print "finish epoch:", epoch

    # 予測クラスと真のクラスを表示する
    print "predict_class:", predict_class
    print "t:", t_train

    # wの可視化
    print "|w_best|:", np.linalg.norm(w_best)
    print "w_best:", w_best
