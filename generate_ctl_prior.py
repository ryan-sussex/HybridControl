#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:27:28 2024

@author: pzc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


W = np.array([[-0.1823255 , -0.32050078],
              [-0.94378347,  0.24966832]])
r = np.array([0.06502312, 0.46049296])


x_init = np.array([1,1]) # initialise x
lr = 0.1 
x = x_init
n_steps = 1000

x_hist = []
x_hist.append(x_init)

# optimize x to maximise softmax(x) for z = 0
for i in range(n_steps):
  a = np.dot(W, x)+ r
  b = np.dot(W[:,0], x)+ r[0]
  partial_z = np.dot(softmax(b) * (np.array([1,0])-softmax(a)), W)
  x_new = x + lr*partial_z
  x = x_new
  x_hist.append(x)
x_hist = np.array(x_hist)


plt.plot(x_hist)
plt.show()


# check whether this makes sense as the max x 
print(softmax(x_init))
print(softmax(x_hist[-1]))
  


