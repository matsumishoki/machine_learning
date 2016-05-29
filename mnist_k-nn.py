# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:45:27 2016

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
from pandas import Series
from sklearn.metrics import f1_score


def kNN(x_train, t_train, x_valid):

    # 正規化2ノルムで割る
    norm_train = np.linalg.norm(x_train, ord=2, axis=1)
    normalized_train_X = x_train / norm_train[:, np.newaxis]
    norm_test = np.linalg.norm(x_valid, ord=2, axis=1)
    normalized_test_X = x_valid / norm_test[:, np.newaxis]
    norm_dev = np.linalg.norm(dev_x, ord=2, axis=1)
    normalized_dev_X = dev_x / norm_dev[:, np.newaxis]

    # f値によってkを決定する
    f1 = []
    for k in [3, 5, 10, 50]:
        pred = []
        n = 100 if len(normalized_test_X) >= 100 else len(normalized_test_X)
        for i in range(n):
            score = np.dot(normalized_dev_X, normalized_test_X[i])
            most_similar = sorted([[s, l] for s,
                                   l in zip(score, dev_y)], reverse=True)
            ranking = most_similar[:k]
            pred.append(Series(map(lambda x: x[1],
                                   ranking)).value_counts().index[0])
        f1.append(f1_score(t_valid[:n], pred, average='macro'))
    k = [3, 5, 10, 50][np.argmax(f1)]

    # コサイン距離が1に近いものをランキングにして、k近傍法を使う

    N = len(x_valid)
    pred_y = []
    for i in range(N):
        score = np.dot(normalized_train_X, normalized_test_X[i])
        most_similar = sorted([[s, l] for s,
                               l in zip(score, t_train)], reverse=True)
        ranking = most_similar[:k]
        pred_y.append(Series(map(lambda x: x[1],
                                 ranking)).value_counts().index[0])
        # valuecount順にならべて一番大きい値をとるindexをもとめる

    return pred_y

if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    x_train, dev_x, t_train, dev_y = train_test_split(
        x_train, t_train, test_size=0.2)

    num_train = len(x_train)
    num_valid = len(x_valid)
    num_test = len(x_test)

    pred_y = kNN(x_train[:10000], t_train, x_valid[:300])
    print "score:", f1_score(t_valid[:len(pred_y)], pred_y, average='macro')
    print "pred_y:", pred_y
