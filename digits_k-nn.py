# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:52:14 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier


class K_NN(object):
    def __init__(self, learning_rate=0.5, max_iteration=100):
        pass

    def fit(self, X, y):
        pass

    def Kneighbors(self, X, n_neighbors=None, return_distance=True):
        pass

    def Kneighbors_graph(self, X=None, n_neighbors=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        pass

if __name__ == '__main__':

    digits = load_digits(2)
    X = digits.data
    num_sumples = len(X)
    T = digits.target

    # ライブラリを使用しない時
    classifier = K_NN()

    # ライブラリを使用した時
    lib_classifier = KNeighborsClassifier()
    lib_classifier.fit(X, T)
    lib_y = lib_classifier.predict(X)
    print "lib_y:", lib_y
    print "lib_classifier_accuracy:", lib_classifier.score(X, T)
