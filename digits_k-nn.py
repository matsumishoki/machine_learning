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
        self.max_iteration = max_iteration

    def fit(self, X, y):
        distance_history = []
        distance_best = 1000
        distance_best_num = []
        x_i_best_num = []
#        num_classes = len(np.unique(y))
#        count = 1
        num_samples = len(X)
        input_data_number = np.random.choice(num_samples)
        for n in range(num_samples):
            distance = np.sum((X[input_data_number, :] - X[n, :])**2)
#            print "distance", distance
            distance_history.append(distance)
#            print "distance_history", distance_history
            k = np.sort(distance_history)
            print "k", k
            if distance < distance_best and distance != 0:
                distance_best = distance
                distance_best_num.append(distance)
                x_i_best_num.append(n)
                print "distance_best", distance_best
        print "x_i_best_num:", x_i_best_num
        print "distance_best_num:", distance_best_num
        print "input_data_number:", input_data_number
        print "distance_best:", distance_best


#        for (x_i, y_i) in zip(X, y):
#            print "count", count
#            print "x_i", x_i
#            print "y_i", y_i
#            count = count + 1
#            if count ==360:
#                plt.matshow(x_i.reshape(8, 8), cmap=plt.cm.gray)

    def Kneighbors(self, X, n_neighbors=None, return_distance=True):
        pass

    def Kneighbors_graph(self, X=None, n_neighbors=None):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        predict_y = self.predict(X)
        correct_rate = np.mean(predict_y == y)
        return correct_rate

if __name__ == '__main__':

    digits = load_digits(2)
    X = digits.data
    num_sumples = len(X)
    T = digits.target

    # ライブラリを使用しない時
    classifier = K_NN()
    classifier.fit(X, T)

    # ライブラリを使用した時
    lib_classifier = KNeighborsClassifier()
    lib_classifier.fit(X, T)
    lib_y = lib_classifier.predict(X)
    print "lib_y:", lib_y
    print "lib_classifier_accuracy:", lib_classifier.score(X, T)
