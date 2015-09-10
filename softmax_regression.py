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
    x = X_raw[0]
    X = np.hstack((X_raw, np.ones((num_examples, 1))))

    # ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)


    # wを10(0～9)クラス定義する（65個の個数を持った配列をrandomで作成する）


    # 確率的勾配降下法


        # 負の対数尤度関数の値を表示する


        # 正解クラスと予測クラスとの比較


    # 予測クラスと真のクラスを表示する


    # wの可視化