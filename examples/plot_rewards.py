#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:38:05 2024

@author: pzc
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# def plot_r(av, std):
#     plt.plot(np.arange(av.shape[0]), av)
#     plt.fill_between(np.arange(av.shape[0]), av - std, av + std, color='b', alpha=0.2)
#     plt.xlabel('Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average reward')
#     plt.legend()
#     plt.xlim(0, 10)
#     plt.ylim(-10, 100)
#     plt.show()
    
# def plot_reward(models_data):
#     colors = ['b', 'g', 'r']  # Colors for the three models
#     labels = ['Model 1', 'Model 2', 'Model 3']  # Labels for the legend

#     for i, (av, std) in enumerate(models_data):
#         plt.plot(np.arange(av.shape[0]), av, label=labels[i], color=colors[i])
#         plt.fill_between(np.arange(av.shape[0]), av - std, av + std, color=colors[i], alpha=0.2)
    
#     plt.xlabel('Episodes')
#     plt.ylabel('Reward')
#     plt.title('Average reward')
#     plt.legend()
#     plt.xlim(0, 10)
#     plt.ylim(-10, 100)
#     plt.show()

# def average_data(file_paths):
#     #data = [np.load(file) for file in file_paths]
#     data = np.stack([np.load(file) for file in file_paths])
#     av_data = np.mean(data, axis=0)
#     std_data = np.std(data, axis=0)
#     return av_data, std_data

# num_runs = 6
# save_dir = "MC_data"
# model = "/AC"

# file_paths = []

# for i in range(num_runs):
#     file_paths.append(save_dir+ model+f"{model}_{i}.npy")
    
    
# av, std = average_data(file_paths)

# plot_r(av,std)


def plot_reward(models_data):
    colors = ['b', 'g']  # Colors for the models
    labels = ['AC', 'SAC_2Q']  # Labels for the legend

    for i, (av, std) in enumerate(models_data):
        plt.plot(np.arange(av.shape[0]), av, label=labels[i], color=colors[i])
        plt.fill_between(np.arange(av.shape[0]), av - std, av + std, color=colors[i], alpha=0.2)
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Average reward')
    plt.legend()
    plt.xlim(0, 50)
    plt.ylim(-10, 100)
    plt.show()
    
def plot_reward(models_data):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']   # Colors for the models
    labels = ['AC', 'SAC', 'HHA']  # Labels for the legend
    linestyles = ['-', '-', '-']  # Line styles for differentiation
    lines = [] 
    plt.figure(figsize=(4, 3))  # Set figure size
    for i, (av, std) in enumerate(models_data):
        line, = plt.plot(np.arange(av.shape[0]), av, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
        plt.fill_between(np.arange(av.shape[0]), av - std, av + std, color=colors[i], alpha=0.2)
        lines.append(line)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    # plt.legend(fontsize=12)
    plt.legend(handles=[lines[2], lines[1], lines[0]], fontsize=12)
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
   
   
    #plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 20)
    plt.ylim(-10, 100)
    plt.xticks(fontsize=12)
    plt.yticks([0, 50,100], labels=['0','50', '100'], fontsize=12)
    plt.tight_layout()
    plt.savefig('average_reward_plot.png', dpi=300)  # Save the figure
    plt.show()
    
# def plot_reward(models_data):
#     colors = ['b', 'g', 'r']  # Colors for the models
#     labels = ['AC', 'SAC_2Q', 'HHA']  # Labels for the legend

#     plt.figure(figsize=(4, 3))  # Set figure size to be small and compact
#     for i, (av, std) in enumerate(models_data):
#         plt.plot(np.arange(av.shape[0]), av, label=labels[i], color=colors[i], linewidth=1.5)
#         plt.fill_between(np.arange(av.shape[0]), av - std, av + std, color=colors[i], alpha=0.3)
    
#     plt.xlabel('Million Steps', fontsize=10)
#     plt.ylabel('Normalized Reward', fontsize=10)
#     plt.title('Gripper', fontsize=12)
#     plt.legend(fontsize=8)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.xticks([0, 100], labels=['0', '100'], fontsize=8)
#     plt.yticks([0, 100 ], fontsize=8)
#     plt.xlim(0, 15)
#     plt.ylim(-10, 100)
#     plt.tight_layout()
#     plt.savefig('average_reward_plot.png', dpi=300)  # Save the figure
#     plt.show()

def average_data(file_paths):
    data = np.stack([np.load(file)[:21] for file in file_paths])
    av_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    return av_data, std_data

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
plot_reward(models_data)