import os
import numpy as np
import importlib
import mountain_car
from hybrid_control.plotting.utils import *


def average_data(file_paths):
    data = [np.load(file) for file in file_paths]
    av_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    return av_data, std_data


def average_rewards(file_paths):
    data = [np.load(file) for file in file_paths]
    min_length = min(len(arr) for arr in data)
    trimmed_rewards = [arr[:min_length] for arr in data]
    
    avg_rewards = np.mean(trimmed_rewards, axis=0)
    
    return avg_rewards
    
    
def plot_coverage_average(obs):
    obs_all = np.squeeze(obs)
    plt.scatter(obs_all[:, 0], obs_all[:, 1], s=0.1)
    
    plt.xlabel('Position', fontsize=22)
    plt.ylabel('Velocity', fontsize=22)
    plt.xlim(-1.3, 0.6)
    plt.ylim(-0.07, 0.07)
    ax = plt.gca()
    
    # Set a fainter grey background color
    ax.set_facecolor('#f0f0f0')  # Lighter shade of grey
    
    # Customize gridlines
    ax.grid(True, which='both', color='white', linewidth=2)
    
    # Adjust the grid z-order to be behind the scatter plot
    ax.set_axisbelow(True)
    
    # Remove the top and right spines (the black box)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set thinner spines for bottom and left
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', labelsize=18)
    
    plt.show()
    

num_runs = 15

save_dir = "MC_data/HHA"
os.makedirs(save_dir, exist_ok=True)

file_paths_rewards = []
file_paths_obs = []

for i in range(num_runs):
    obs, rewards = mountain_car.main()

    np.save(os.path.join(save_dir, f"obs_{i}.npy"), obs)
    np.save(os.path.join(save_dir, f"HHA_{i}.npy"), rewards)
    
    file_paths_obs.append(save_dir+f"/obs_{i}.npy")
    file_paths_rewards.append(save_dir+f"/HHA_{i}.npy")

# av_rewards, std_rewards = average_data(file_paths_rewards)
# av_obs, std_obs = average_data(file_paths_obs)

# plot_av_rewards(av_rewards, std_rewards)
# plot_coverage(av_obs)