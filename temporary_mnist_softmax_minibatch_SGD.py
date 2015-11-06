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


# PRML pp209 l1 交差エントロピー誤差と正解率を求める関数
def error_and_accuracy(w, X, t):
    # 交差エントロピー誤差を計算する
    y = softmax(np.inner(X, w))
    T = onehot(t)
    num_examples = len(X)
    error = np.sum(-(T*(np.log(y)))) / num_examples
    assert not np.any(np.isnan(error))
    assert not np.any(np.isinf(error))

    # 正解率を計算する
    predict_class = np.argmax(y, axis=1)
    num_correct = np.sum(t == predict_class)
    accuracy = num_correct / float(num_examples) * 100

    return (error, accuracy)

# main文
if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape
    print "x_valid.shape:", x_valid.shape
    print "t_valid.shape:", t_valid.shape
    print "x_test.shape:", x_test.shape
    print "t_test.shape:", t_test.shape

    num_train = len(x_train)
    num_valid = len(x_valid)
    num_test = len(x_test)

    # wxの中に定数項であるバイアス項を入れ込む
    X_train = np.hstack((x_train, np.ones((num_train, 1))))
    X_valid = np.hstack((x_valid, np.ones((num_valid, 1))))
    X_test = np.hstack((x_test, np.ones((num_test, 1))))

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = X_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.001  # learning_rate(学習率)を定義する
    max_iteration = 200      # 学習させる回数
    w_scale = 0.001        # wのノルムの大きさを調整する変数
    batch_size = 300       # ミニバッチ1つあたりのサンプル数

    # dim_features次元の重みをnum_classesクラス分用意する
    w = w_scale * np.random.randn(num_classes, dim_features)

    error_history = []
    train_accuracy_history = []
    error_valid_history = []
    valid_accuracy_history = []

    w_best = 0
    valid_accuracy_best = 0
    valid_error_best = 10
    num_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size

    for epoch in range(max_iteration):
        print "epoch:", epoch

        time_start = time.time()
        perm = np.random.permutation(num_train)

        for batch_indexes in np.array_split(perm, num_batches):
            X_batch = X_train[batch_indexes]
            t_batch = t_train[batch_indexes]
            y_batch = softmax(np.inner(X_batch, w))
            T_batch = onehot(t_batch)
            grad_w = np.dot((y_batch - T_batch).T, X_batch)
            w = w - learning_rate * grad_w

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 訓練データセットの交差エントロピー誤差と正解率を表示する
        train_error, train_accuracy = error_and_accuracy(w, X_train, t_train)
        print "[train] Error:", train_error
        print "[train] Accuracy:", train_accuracy
        error_history.append(train_error)
        train_accuracy_history.append(train_accuracy)

        # 検証データセットの交差エントロピー誤差と正解率を表示する
        valid_error, valid_accuracy = error_and_accuracy(w, X_valid, t_valid)
        print "[valid] Error:", valid_error
        print "[valid] Accuracy:", valid_accuracy
        error_valid_history.append(valid_error)
        valid_accuracy_history.append(valid_accuracy)

        print "|w|:", np.linalg.norm(w)

        # 学習曲線をプロットする
        plt.plot(error_history, label="error")
        plt.plot(error_valid_history, label="valid_error")
        plt.title("error")
        plt.legend(['train_error', 'valid_error'])
        plt.grid()
        plt.show()

        plt.plot(train_accuracy_history, label="train_accuracy")
        plt.plot(valid_accuracy_history, label="valid_accuracy")
        plt.legend(loc="lower right")
        plt.title("train_accuracy")
        plt.grid()
        plt.show()

        # 検証データのerror率が良ければwの値を保存し，wの最善値を格納する
        if valid_error <= valid_error_best:
            w_best = w
            epoch_best = epoch
            valid_error_best = valid_error
            valid_accuracy_best = valid_accuracy
            print "epoch_best:", epoch_best
            print "valid_error_best:", valid_error_best
            print "valid_accuracy_best:", valid_accuracy_best

    # 学習済みのモデルでテストセットを評価して正解率を求める
    test_error, test_accuracy = error_and_accuracy(w, X_test, t_test)

    print
    print "[test]  Accuracy:", test_accuracy
    print "[valid] Accuracy (best)  :", valid_accuracy_best
    print "[valid] Error (best):", valid_error_best
    print "Best epoch :", epoch_best
    print "Finish epoch:", epoch
    print "Batch size:", batch_size
    print "Learning rate:", learning_rate
    print

    # wの可視化
    print "|w_best|:", np.linalg.norm(w_best)
    print "w_best:", w_best
    fig, axes = plt.subplots(2, 5,  figsize=(10, 4))
    for w_k, ax in zip(w_best, axes.ravel()):
        w_true = w_k[0:784]  # w_trueとは結果をプロットするために定義したものである
        ax.matshow(w_true.reshape(28, 28), cmap=plt.cm.gray)
    plt.show()
