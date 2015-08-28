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
    plt.matshow(images[5], cmap=plt.cm.gray)
    plt.show()
    """ρを定義する(ρ=0.5で良いか判断し，収束しなければ値を変える．)"""
    p = 0.5
    """1. wを定義する（64個の個数を持った配列をrandomで作成する）"""
    w=[]

    """5. 全て正しく識別できるまで繰り返す．"""
        """4. for(i=0; i< nun_examples; i++)"""
            """2. Xのx_iを取り出す．"""

            """2. g(x_i) = <w, x_i>を計算する"""

            """3. もしｘ_iが間違っていたならば，wを修正する(w_new = w + ρ*t_i*x_i )
               間違っているとは，g(x_i)の符号がクラスラベルt_iと逆の場合である """

    """結果を表示する"""
    """結果とは, g(x_i) = <w_new, x_i>の直線を描き，tをプロットしたものである"""