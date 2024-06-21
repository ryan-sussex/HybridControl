#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:38:05 2024

@author: pzc
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from hybrid_control.plotting.utils import *

# Define the models and their directories
models = ['AC', 'SAC_2Q', 'HHA']
save_dir = "MC_data"

# List to store average and std data for each model
models_data = []

num_runs = 6  # Number of runs/files per model

for model in models:
    file_paths = [os.path.join(save_dir, model, f"{model}_{i}.npy") for i in range(num_runs)]
    av, std = average_data(file_paths)
    models_data.append((av, std))

# Plot the results
plot_rewards_overall(models_data)