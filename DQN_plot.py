#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:51:40 2024

@author: pzc
"""
from hybrid_control.plotting.utils import *

import numpy as np

a = np.load('examples/a.npy')
b = np.load('examples/b.npy')

combined_array = np.stack((a, b), axis=1)

plot_coverage(combined_array)