#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:27:28 2024

@author: pzc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


def generate_prior(
    W, r, x_init, z, lr: float = 1.0, n_steps: int = 10000, threshold: float = 0.9
):
    """

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
    threshold : float
        Threshold for probability maximization.
    z : INT
         max_x P(Z=z|x)

    Returns
    -------
    x_hist : array
        optimized xs over time.

    """

    x = x_init
    x_hist = []
    x_hist.append(x_init)

    # optimize x to maximise softmax(x) for z = z
    for i in range(n_steps):

        # add threshold
        if softmax(W @ x_hist[-1] + r)[z] > threshold:  # what if z > num x_states?
            return x_hist

        a = np.dot(W, x) + r
        P = softmax(a)
        ez = np.eye(P.shape[0])[z]
        partial_P = W.T @ (ez - P)

        x_new = x + lr * partial_P
        x = x_new
        x_hist.append(x)

    return x_hist


def generate_all_priors(
    W: np.ndarray,
    r: np.ndarray,
    lr: float = 1.0,
    n_steps: int = 10000,
    threshold: float = 0.9,
):
    """

    Parameters
    ----------
    W : ARRAY
        Weight matrix of LogReg
    r : array
        Biases of LogReg.
    lr : INT
        learning rate.
    n_steps : INT
        Num optimization steps.
    threshold : float
        Threshold for probability maximization.

    Returns
    -------
    priors : LIST[array]
        optimized xs over time.

    """
    x_init = np.zeros(W.shape[1])
    priors = []
    for i in range(W.shape[0]):
        opt_hist = generate_prior(
                W, r, x_init, z=i, lr=lr, n_steps=n_steps, threshold=threshold)
        prior = opt_hist[-1]
        priors.append(prior)
    return priors
