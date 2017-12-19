# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:33:08 2017

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers
from chainer import cuda
from chainer import Variable, Chain, optimizers
from chainer.cuda import cupy


class ConvNet(Chain):
        def __init__(self):
            super(ConvNet, self).__init__(
                    conv_1 = L.Convolution2D(1, 50, 5),
                    conv_12 = L.Convolution2D(50, 50, 1),
                    conv_2 = L.Convolution2D(50, 100, 5),
                    conv_3 = L.Convolution2D(100, 200, 4),
                    l_1 = L.Linear(200, 400),
                    l_2 = L.Linear(400, 10),
                    )
        def loss_and_accuracy(self, x_data, t_data, train):
            x = Variable(x_data.reshape(-1, 1, 28, 28))
            t = Variable(t_data)
            h = self.conv_1(x)
            h = self.conv_12(h)
            h = F.max_pooling_2d(h, 2)
            h = F.relu(h)
            h = self.conv_2(h)
            h = F.max_pooling_2d(h, 2)
            h = F.relu(h)
            h = self.conv_3(h)
            h = F.relu(h)
            h = self.l_1(h)
            h = F.relu(h)
            y = self.l_2(h)
#            loss = F.softmax_cross_entropy(a_y, t)
#            accuracy = F.accuracy(a_y, t)
#            return loss, cuda.to_cpu(accuracy.data) * 100
            accuracy = F.accuracy(y, t)
#            print("acuuracy",accuracy.data*100)
            return F.softmax_cross_entropy(y, t), accuracy * 100

def loss_and_accuracy_average(model, x_data, t_data, num_batches, train):
    accuracies = []
    losses = []
    total_data = np.arange(len(x_data))
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(x_data[indexes])
        T_batch = cuda.to_gpu(t_data[indexes])
        loss, accuracy = model.loss_and_accuracy(X_batch, T_batch, train)
        accuracy_cpu = cuda.to_cpu(accuracy.data)
        loss_cpu = cuda.to_cpu(loss.data)
        accuracies.append(accuracy_cpu)
        losses.append(loss_cpu)
    return np.mean(accuracies), np.mean(losses)

if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print ("x_train.shape:", x_train.shape)
    print ("t_train.shape:", t_train.shape)

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    print ("x_train.shape:", x_train.shape)
    print ("t_train.shape:", t_train.shape)
    print ("x_valid.shape:", x_valid.shape)
    print ("t_valid.shape:", t_valid.shape)
    print ("x_test.shape:", x_test.shape)
    print ("t_test.shape:", t_test.shape)

    num_train = len(x_train)
    num_valid = len(x_valid)
    num_test = len(x_test)
    
    # 訓練用データのshapeを(n_samples, channel, hight, width)に変更する
#    x_train = x_train.reshape(num_train,1,28,28)
#    x_valid = x_valid.reshape(num_valid,1,28,28)

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.01  # learning_rate(学習率)を定義する
    max_iteration = 100      # 学習させる回数
    batch_size = 200       # ミニバッチ1つあたりのサンプル数

    model = ConvNet().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    loss_train_history = []
    train_accuracy_history = []
    loss_valid_history = []
    valid_accuracy_history = []

    valid_accuracy_best = 0
    valid_loss_best = 10
    num_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size

    # 学習させるループ
    for epoch in range(max_iteration):
        print ("epoch:", epoch)
        w_1_grad_norms = []
        w_2_grad_norms = []
        w_3_grad_norms = []
        b_1_grad_norms = []
        b_2_grad_norms = []
        b_3_grad_norms = []

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        perm = np.random.permutation(num_train)

        for batch_indexes in np.array_split(perm, num_batches):
            x_batch = cuda.to_gpu(x_train[batch_indexes])
            t_batch = cuda.to_gpu(t_train[batch_indexes])

            # 勾配を初期化する            
            model.zerograds()

            # 順伝播を計算し、誤差と精度を取得
            batch_loss, batch_accuracy = model.loss_and_accuracy(x_batch,
                                                                 t_batch, True)
            # 逆伝播
            batch_loss.backward()
            optimizer.update()
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print ("time_elapsed:", time_elapsed)
        
        # 訓練データセットの交差エントロピー誤差と正解率を表示する
        train_accuracy, train_loss = loss_and_accuracy_average(
                model, x_train, t_train, num_batches, False)
        train_accuracy_history.append(train_accuracy)
        loss_train_history.append(train_loss)
        print ("[train] Loss:", train_loss)
        print ("[train] Accuracy:", train_accuracy)
        
        # 検証用データセットの交差エントロピー誤差と正解率を表示する
        valid_accuracy, valid_loss = loss_and_accuracy_average(
                model, x_valid, t_valid, num_batches, False)
        valid_accuracy_history.append(valid_accuracy)
        loss_valid_history.append(valid_loss)
        print ("[valid] Loss:", valid_loss)
        print ("[valid] Accuracy:", valid_accuracy)
        
        # 学習曲線をプロットする
        # plot learning curves
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(loss_train_history)
        plt.plot(loss_valid_history)
        plt.legend(["train", "valid"], loc="best")
        plt.ylim([0.0, 0.2])
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(train_accuracy_history)
        plt.plot(valid_accuracy_history)
        plt.legend(["train", "valid"], loc="best")
        plt.ylim([97, 100])
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        plt.draw()
        
        # 検証データの誤差が良ければwの最善値を保存する
        if valid_loss <= valid_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            valid_loss_best = valid_loss
            valid_accuracy_best = valid_accuracy
            print ("epoch_best:", epoch_best)
            print ("valid_loss_best:", valid_loss_best)
            print ("valid_accuracy_best:", valid_accuracy_best)
            
    # 検証データセットの交差エントロピー誤差と正解率を表示する
    test_accuracy, test_loss = loss_and_accuracy_average(
        model_best, x_test, t_test, num_batches, False)

    print ("[test]  Accuracy:", test_accuracy)
    print ("[valid] Accuracy (best)  :", valid_accuracy_best)
    print ("[valid] Loss (best):", valid_loss_best)
    print ("[train] Loss:", train_accuracy_history)
    print ("Best epoch :", epoch_best)
    print ("Finish epoch:", epoch)
    print ("Batch size:", batch_size)
    print ("Learning rate:", learning_rate)
    print ("|w_1_best|:", np.linalg.norm(cuda.to_cpu(model.l_1.W.data),
                                  axis=0).mean())
    print ("|w_2_best|:", np.linalg.norm(cuda.to_cpu(model.l_2.W.data),
                                  axis=0).mean())
