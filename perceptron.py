# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:19:34 2015

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# main文
if __name__=='__main__':
    digits = load_digits(2)
    X = digits.data
    t = digits.target
    t[t==0] = -1
    nun_examples = len(X)
    images = digits.images
    plt.matshow(images[4], cmap=plt.cm.gray)
    plt.show()
    """ρを定義する(ρ=0.5で良いか判断し，収束しなければ値を変える．)"""
    p = 0.5
    """1. wを定義する（64個の個数を持った配列をrandomで作成する）"""
    w = np.random.randn(1, 64)

    """5. 全て正しく識別できるまで繰り返す．"""
        #"""4. for(i=0; i< nun_examples; i++)"""
    for x_i, t_i in zip(X, t):
         #   """2. Xのx_iと、t_iを取り出す．"""
        #print "x_i:", x_i
        #print "t_i:", t_i
          #  """2. g(x_i) = <w, x_i>を計算する"""
        g_i = np.inner(w, x_i)
        #print "g_i:", g_i
           # """3. もしｘ_iが間違っていたならば，wを修正する(w_new = w + ρ*t_i*x_i )
        
          #    間違っているとは，g(x_i)の符号がクラスラベルt_iと逆の場合である """
        if t_i * g_i < 0:
            w_new = w + p * t_i * x_i
            g_i_new = np.inner(w_new, x_i)
            if t_i * g_i_new < 0:
                print "no"
            else:
                print "good"
                print "t_i * g_i_new:", t_i * g_i_new
        else:
            print "Ok"
    """結果を表示する
    結果とは, g(x_i) = <w_new, x_i>の直線を描き，tをプロットしたものである"""
    """具体的な結果の出力の手順としては
    1. if t_i * g_i > 0 or t_i * g_i_new > 0:の条件文を記述する
    2. if w or w_new:により正しいwとw_newを取り出す
    3. plt.matshow()を使い、wとw_newの点をplotする(線を引きたい)
    4. X[0]からX[360]までの点をplotする
    5.可視化が出来て問題がなければ完成である"""