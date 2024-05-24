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


num_runs = 3

save_dir = "MC_data"
os.makedirs(save_dir, exist_ok=True)

file_paths_rewards = []
file_paths_obs = []

for i in range(num_runs):
    obs, rewards = mountain_car.main()

    np.save(os.path.join(save_dir, f"obs_run_{i}.npy"), obs)
    np.save(os.path.join(save_dir, f"rewards_run_{i}.npy"), rewards)
    
    file_paths_obs.append(save_dir+f"/obs_run_{i}.npy")
    file_paths_rewards.append(save_dir+f"/rewards_run_{i}.npy")

av_rewards, std_rewards = average_data(file_paths_rewards)
av_obs, std_obs = average_data(file_paths_obs)

plot_av_rewards(av_rewards, std_rewards)
plot_av_coverage(av_obs)