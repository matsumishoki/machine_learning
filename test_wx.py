# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:06:29 2015

@author: matsumi
"""

import numpy as np

w = np.zeros((10, 785))
x_batch = np.ones((200, 785))

a = np.inner(w, x_batch)
print a.shape
