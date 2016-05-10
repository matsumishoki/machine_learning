# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:52:14 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class K_NN(object):
    def __init__(self, learning_rate=0.5, max_iteration=100):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration

    def fit(self, X, y):
        dimension = X.shape[1]
        w = np.random.randn(dimension)
        y = self._convert_label(y)
        for i in range(self.max_iteration):

            for (x_i, y_i) in zip(X, y):
                g_i = np.inner(w, x_i)

                if g_i * y_i < 0:
                    w = w + self.learning_rate * y_i * x_i

        # wをインスタンス変数として持たせる
        self.w = w

    def predict(self, X):
        y = np.sign(np.inner(self.w, X))
        return self._revert_label(y)

    def score(self, X, y):
        predict_y = self.predict(X)
        correct_rate = np.mean(predict_y == y)
        return correct_rate

    def _convert_label(self, y):
        y = y.copy()
        y[y == 0] = -1
        return y

    def _revert_label(self, y):
        y = y.copy()
        y[y == -1] = 0
        return y

if __name__ == '__main__':

    digits = load_digits(2)
    X = digits.data
    num_sumples = len(X)
    T = digits.target

    classifier = K_NN()

    classifier.fit(X, T)

    y = classifier.predict(X)

    print classifier.score(X, T)
    accuracy = np.sum(y == T)/float(num_sumples)
    print y
    print "accuracy:", accuracy
