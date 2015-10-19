# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:22:18 2015

@author: matsumi
"""

import numpy as np

w = np.zeros((10, 785))
x_i = np.arange(785)

a = np.inner(w, x_i)
print a.shape

y_i = np.ones((10,))
T = np.ones((10,))
o = np.expand_dims(y_i - T, 1)
print o.shape
