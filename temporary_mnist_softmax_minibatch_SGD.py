# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:54:30 2015

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time


def softmax(s):
    len(s.shape)
    exp_s = np.exp(s)
    if len(s.shape) == 1:
        return exp_s / np.sum(exp_s)

    if len(s.shape) == 2:
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def onehot(k, num_classes=10):
    assert isinstance(k, np.ndarray)
    assert k.ndim == 1
    assert k.dtype == np.int, k.dtype

    num_examples = len(k)
    t_onehot = np.zeros((num_examples, num_classes))
    t_onehot[np.arange(num_examples), k] = 1
    return t_onehot


# main文
if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    # t_train = map(int, t_train)
    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    v = train_test_split(x_train, t_train, test_size=0.1, random_state=100)
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
    learning_rate = 0.001

    # 収束するまで繰り返す
    max_iteration = 200

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
    total_valid_error_best = 10
    batch_size = 200                      # ミニバッチ1つあたりのサンプル数
    num_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size

    for epoch in range(max_iteration):
        print "epoch:", epoch

        time_start = time.time()
        perm = np.random.permutation(num_train)
#        for i in perm:
#            x_i = X_train[i]
#            t_i = t_train[i]
#            y_i = softmax(np.inner(w, x_i))
#            T = onehot(t_i)
#            w_new = w - learning_rate * np.expand_dims(y_i - T, 1) * x_i
#            w = w_new
        for batch_indexes in np.array_split(perm, num_batches):
            X_batch = X_train[batch_indexes]
            t_batch = t_train[batch_indexes]
            y_batch = softmax(np.inner(X_batch, w))
            T_batch = onehot(t_batch)
            grad_w = np.dot((y_batch - T_batch).T, X_batch)
            w_new = w - learning_rate * grad_w
            w = w_new

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 訓練セットの負の対数尤度関数の値を表示する
        errors = []
        for batch_indexes in np.array_split(np.arange(num_train), num_batches):
            X_batch = X_train[batch_indexes]
            t_batch = t_train[batch_indexes]
            y_batch = softmax(np.inner(X_batch, w))
            T_batch = onehot(t_batch)
            error = np.sum(-(T_batch*(np.log(y_batch)))) / batch_size
            errors.append(error)
            assert not np.any(np.isnan(error))
            assert not np.any(np.isinf(error))

        total_error = np.mean(errors)
        print "error:", total_error
        error_history.append(total_error)

        # 検証セットの負の対数尤度関数の値を表示する
        valid_errors = []
        for batch_indexes in np.array_split(np.arange(num_valid),
                                            num_valid_batches):
            X_batch = X_valid[batch_indexes]
            t_batch = t_valid[batch_indexes]
            y_batch = softmax(np.inner(X_batch, w))
            T_batch = onehot(t_batch)
            valid_error = np.sum(-(T_batch*(np.log(y_batch)))) / batch_size
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
            correct_valid_percent_best = correct_valid_percent
            print "correct_valid_percent_best:", correct_valid_percent_best
            epoch_best = epoch
            print "epoch_best:", epoch_best
            print "valid_error_best:", total_valid_error_best

    # 学習済みのモデルでテストセットを評価して正解率を求める
    y_test = softmax(np.inner(X_test, w_best))
    predict_class_test = np.argmax(y_test, axis=1)
    num_correct_test = np.sum(t_test == predict_class_test)
    correct_test_percent = num_correct_test / float(num_test) * 100
    print "learning_rate:", learning_rate
    print "valid_error_best:", total_valid_error_best
    print "correct_valid_percent_best:", correct_valid_percent_best
    print "epoch_best:", epoch_best
    print "finish epoch:", epoch
    print "correct_test_percent:", correct_test_percent

    # 予測クラスと真のクラスを表示する
    print "predict_class:", predict_class
    print "t:", t_train

    # wの可視化
    print "|w_best|:", np.linalg.norm(w_best)
    print "w_best:", w_best
    for w_k in w_best:
        w_true = w_k[0:784]  # w_trueとは結果をプロットするために定義したものである
        # print "w_true:", w_true
        plt.matshow(w_true.reshape(28, 28), cmap=plt.cm.gray)
        plt.show()
