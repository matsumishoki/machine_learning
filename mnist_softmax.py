# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:19:13 2015

@author: matsumi
"""

import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

x_train, t_train, x_test, t_test = load_mnist.load_mnist()

# main文

# データ・セットの読み込み

# ρを定義する(ρ=0.1で良いか判断し，収束しなければ値を変える．)

# 収束するまで繰り返す

# dim_features次元の重みをnum_classesクラス分用意する

# 確率的勾配降下法

# 負の対数尤度関数の値を表示する

# 正解クラスと予測クラスとの比較

# 学習曲線をプロットする

# 予測クラスと真のクラスを表示する

# wの可視化
