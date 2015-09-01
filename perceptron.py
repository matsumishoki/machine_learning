# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:19:34 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# main文
if __name__ == '__main__':
    digits = load_digits(2)
    images = digits.images
    plt.matshow(images[4], cmap=plt.cm.gray)
    plt.show()

    # データ・セットの読み込み
    X = digits.data
    t = digits.target
    t[t == 0] = -1
    num_examples = len(X)

    # ρを定義する(ρ=0.5で良いか判断し，収束しなければ値を変える．)
    rho = 0.5
    # 最大の繰り返し回数
    max_iteration = 100

    # 1. wを定義する（64個の個数を持った配列をrandomで作成する）
    w = np.random.randn(64)

    # 5. 全て正しく識別できるまで繰り返す．
    for epoch in range(max_iteration):
        # 2. Xのx_iと、t_iを取り出す．
        for x_i, t_i in zip(X, t):
            g_i = np.inner(w, x_i)
            # 3. もしｘ_iが間違っていたならば，wを修正する(w_new = w + ρ*t_i*x_i )
            # 間違っているとは，g(x_i)の符号がクラスラベルt_iと逆の場合である
            if t_i * g_i < 0:
                w_new = w + rho * t_i * x_i
            else:
                w_new = w

            w = w_new

        # 予測クラスと正解ラベルとの比較をし，正解率を表示する
        y = np.sign(np.inner(w, X))  # 予測クラス
        num_correct = np.sum(y == t)  # 正解の個数
        correct_percent = num_correct / float(num_examples) * 100
        print "correct_percent:", correct_percent
        if correct_percent == 100.0:
            break

    # 予測クラスと真のクラスを表示する
    print "y:", y
    print "t:", t

    # wの可視化
    print "w:", w
    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
