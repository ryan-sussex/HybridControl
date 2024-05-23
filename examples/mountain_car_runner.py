import os
import numpy as np
import importlib
import mountain_car
from hybrid_control.plotting.utils import *


num_runs = 5

save_dir = "MC_data"
os.makedirs(save_dir, exist_ok=True)

def load_and_average_arrays(file_paths):
    arrays = [np.load(file) for file in file_paths]
    average_array = np.mean(arrays, axis=0)

    return average_array


file_paths_rewards = []
file_paths_obs = []

for i in range(num_runs):
    obs, rewards = mountain_car.main()

    np.save(os.path.join(save_dir, f"obs_run_{i}.npy"), obs)
    np.save(os.path.join(save_dir, f"rewards_run_{i}.npy"), rewards)

    print(f"Run {i+1} completed and arrays saved.")
    
    file_paths_rewards.append(save_dir+f"/obs_run_{i}.npy")
    file_paths_obs.append(save_dir+f"/rewards_run_{i}.npy")
    

av_rewards = load_and_average_arrays(file_paths_rewards)
av_obs = load_and_average_arrays(file_paths_obs)


plot_total_reward(av_rewards)
plot_coverage(av_obs)