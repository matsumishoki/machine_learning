# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 19:09:38 2015

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


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
    x_train = np.hstack((x_train, np.ones((num_train, 1)))) #(1×)
    x_valid = np.hstack((x_valid, np.ones((num_valid, 1))))
    x_test = np.hstack((x_test, np.ones((num_test, 1))))

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.001  # learning_rate(学習率)を定義する
    max_iteration = 200      # 学習させる回数
    w_scale = 0.001        # wのノルムの大きさを調整する変数
    batch_size = 300       # ミニバッチ1つあたりのサンプル数

    # dim_features次元の重みをnum_classesクラス分用意する
    # 入力層と中間層の間のw_1(D×M)
    w_1 = w_scale * np.random.randn(num_classes, dim_features)

    # 中間層と出力層の間のw(M×K)
    w_2 = w_scale * np.random.randn(num_classes, dim_features)

    # 学習させるループ
        # mini batchi SGDで重みを更新させるループ
            # 順伝播
            # 入力層と中間層のx(1×D)w_1(D×M)によって訓練データとの行列積(xw_1)を計算する(a_j(1×M)を求める)

            # 求まったa_j(1×M)を隠れユニットのz(1×M)にする(活性化関数にa_j(1×M)を代入する)

            # 入力されたz(1×M)と,中間層と入力層のw_2(M×K)によって行列積(zw_2)を計算する

            # 逆伝播
            # 出力された値y(1×K)から正解ラベルt(1×K)を引く(y-t)(δ_y(1×K))

            # z.T(M×1)とδ_y(1×K)との行列積(z.T δ_y)を計算する(誤差をw_2で微分したもの(grad_w_2(M×K)))

            # δ_y(1×K)とw_2(K×M)との行列積(δ_y w2)を計算する(一時的にgrad_z(1×M))

            # grad_z(1×M)と(1-z**2)の要素積を計算する(δ_z(1×M))

            # x.T(D×1)とδ_z(1×M)との行列積(x.T δ_z)を計算する(grad_w_1(D×M))

            # w_1を更新する(w_1 = w_1 - learning_rate*grad_w_1)

            # w_2を更新する(w_2 = w_2 - learning_rate*grad_w_2)

        # 誤差
        # E(K×K)を出す0.5×(y-t)×(y-t).T次元数は，{0.5×(1×K)(K×1)}
        # 訓練データセットの交差エントロピー誤差と正解率を表示する

        # 検証データセットの交差エントロピー誤差と正解率を表示する

        # 学習曲線をプロットする

        # 検証データの誤差が良ければwの最善値を保存する

    # 学習済みのモデルをテストセットで誤差と正解率を求める

    # wの可視化
