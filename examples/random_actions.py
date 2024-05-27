#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:16:38 2024

@author: pzc
"""

import gym
import numpy as np
from hybrid_control.plotting.utils import *


# Create the Continuous Mountain Car environment
env = gym.make('MountainCarContinuous-v0')

# Set the number of timesteps
num_timesteps = 10000

# Initialize the environment
state = env.reset()

obs = []
# Loop over the timesteps
for t in range(num_timesteps):
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Take the action in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    obs.append(observation)
    # If the episode is done, reset the environment
    if terminated or truncated:
        state = env.reset()
        
    # Optionally, render the environment (comment this out if not needed)
    # env.render()

# Close the environment
env.close()

plot_coverage(obs)