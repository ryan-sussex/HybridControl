#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:27:28 2024

@author: pzc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


def generate_prior(W, r, x_init, lr, n_steps, threshold, z):
    '''

    Parameters
    ----------
    W : ARRAY
        Weight matrix of LogReg
    r : array
        Biases of LogReg.
    x_init : ARRAY
        Initial x value.
    lr : INT
        learning rate.
    n_steps : INT
        Num optimization steps.
    h : INT
        Threshold for probability maximization.
    z : INT
         max_x P(Z=z|x)

    Returns
    -------
    x_hist : array
        optimized xs over time.

    '''
    
    x = x_init

    x_hist = []
    x_hist.append(x_init)

    # optimize x to maximise softmax(x) for z = z
    for i in range(n_steps):
        
        # add threshold
        if softmax(x_hist[-1])[0] > threshold:
            return x_hist
        
        a = np.dot(W, x)+ r
        b = np.dot(W[:,z], x)+ r[z]
        partial_z = np.dot(softmax(b) * (np.eye(2)[z]-softmax(a)), W)
        x_new = x + lr*partial_z
        x = x_new
        x_hist.append(x)
  
    x_hist = np.array(x_hist)
    
    
    return x_hist


W = np.array([[-0.1823255 , -0.32050078],
              [-0.94378347,  0.24966832]])
r = np.array([0.06502312, 0.46049296])

lr = 0.1 
n_steps = 1000
threshold = 0.99
x_init = np.array([1,1]) # initialise x


x_hist = generate_prior(W, r, x_init, lr, n_steps, threshold, 1)

plt.plot(x_hist)
plt.show()


# check whether this makes sense as the max x 
print(softmax(x_init))
print(softmax(x_hist[-1]))
  


