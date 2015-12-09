# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 11:34:50 2015

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import chainer.functions as F
from chainer import Variable, FunctionSet
from chainer.optimizers import SGD, Adam


def error_and_accuracy(model, x_data, t_data):
    x = Variable(x_data)
    t = Variable(t_data)

    a_z = linear_1(x)
    z = F.tanh(a_z)
    a_y = linear_2(z)

    loss = F.softmax_cross_entropy(a_y, t)
    accuracy = F.accuracy(a_y, t)
    return loss.data, accuracy.data * 100

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

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.5  # learning_rate(学習率)を定義する
    max_iteration = 100      # 学習させる回数
    batch_size = 200       # ミニバッチ1つあたりのサンプル数
    dim_hidden = 200         # 隠れ層の次元数を定義する

    linear_1 = F.Linear(dim_features, dim_hidden)
    linear_2 = F.Linear(dim_hidden, num_classes)
    model = FunctionSet(linear_1=linear_1,
                        linear_2=linear_2)
                        
    optimizer = SGD(learning_rate)
    optimizer.setup(model)


    error_history = []
    train_accuracy_history = []
    error_valid_history = []
    valid_accuracy_history = []

    w_1_best = 0
    w_2_best = 0
    valid_accuracy_best = 0
    valid_error_best = 10
    num_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size

    # 学習させるループ
    for epoch in range(max_iteration):
#        print "epoch:", epoch

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        perm = np.random.permutation(num_train)

        for batch_indexes in np.array_split(perm, num_batches):
            x_batch = x_train[batch_indexes]
            t_batch = t_train[batch_indexes]

            x = Variable(x_batch)
            t = Variable(t_batch)

            # 順伝播
            a_z = linear_1(x)
            z = F.tanh(a_z)
            a_y = linear_2(z)
            
            # 逆伝播            
            loss = F.softmax_cross_entropy(a_y, t)
            accuracy = F.accuracy(a_y, t)

            optimizer.zero_grads()
            loss.backward()
            optimizer.update()

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 誤差
        # E(K×K)を出す0.5×(y-t)×(y-t).T次元数は，{0.5×(1×K)(K×1)}
        # E = sum(t×log(y)(1×K))
        # 訓練データセットの交差エントロピー誤差と正解率を表示する
        train_error, train_accuracy = error_and_accuracy(model, 
                                                         x_train, t_train)
        print "[train] Error:", train_error
        print "[train] Accuracy:", train_accuracy
        error_history.append(train_error)
        train_accuracy_history.append(train_accuracy)

        # 検証データセットの交差エントロピー誤差と正解率を表示する
        valid_error, valid_accuracy = error_and_accuracy(model,
                                                         x_valid, t_valid)
        print "[valid] Error:", valid_error
        print "[valid] Accuracy:", valid_accuracy
        error_valid_history.append(valid_error)
        valid_accuracy_history.append(valid_accuracy)

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

        # 検証データの誤差が良ければwの最善値を保存する
        if valid_error <= valid_error_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            valid_error_best = valid_error
            valid_accuracy_best = valid_accuracy
            print "epoch_best:", epoch_best
            print "valid_error_best:", valid_error_best
            print "valid_accuracy_best:", valid_accuracy_best
            print
    # 学習済みのモデルをテストセットで誤差と正解率を求める
    test_error, test_accuracy = error_and_accuracy(model,
                                                   x_test, t_test)

    print "[test]  Accuracy:", test_accuracy
    print "[valid] Accuracy (best)  :", valid_accuracy_best
    print "[valid] Error (best):", valid_error_best
    print "Best epoch :", epoch_best
    print "Finish epoch:", epoch
    print "Batch size:", batch_size
    print "Learning rate:", learning_rate
    print "dim_hidden:", dim_hidden
    print

    # wの可視化
    print "|w_1_best|:", np.linalg.norm(model.linear_1.W.data)
    print "w_1_best:", model.linear_1.W.data
    print "|w_2_best|:", np.linalg.norm(model.linear_2.W.data)
    print "w_2_best:", model.linear_2.W.data
    w_best = np.dot(model.linear_2.W.data, model.linear_1.W.data)
    fig, axes = plt.subplots(2, 5,  figsize=(10, 4))
    for w_k, ax in zip(w_best, axes.ravel()):
        w_true = w_k[0:784]  # w_trueとは結果をプロットするために定義したものである
        ax.matshow(w_true.reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

