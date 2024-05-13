import logging
from typing import List, Callable
import numpy as np
import gym

from ssm import SLDS

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.controller import Controller, get_initial_controller

from hybrid_control.plotting.utils import *

from utils import create_video

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    ENV_STEPS = 1000

    env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env.reset()
    
    K = 5  # would be unknown
    OBS_DIM = 2
    ACT_DIM = 1
    N_ITER = 100
    N_STEPS = 100

    controller = get_initial_controller(OBS_DIM, ACT_DIM, K)
    action = controller.policy()

    obs = []
    actions = []
    frames = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            observation, info = env.reset()

        obs.append(observation)
        actions.append(action)

        action = controller.policy(observation, action)
        
        if i % 200 == 199:
            create_video(frames, 60, "./video/out")
            try:
                controller = controller.estimate_and_identify(np.stack(obs), np.stack(actions))    
            except Exception:
                pass

    create_video(frames, 60, "./video/out")
    env.close()
