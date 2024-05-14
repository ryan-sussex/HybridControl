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
    ENV_STEPS = 2000
    REFIT_EVERY = 500

    env = gym.make('Pendulum-v1', g=9.81, render_mode="rgb_array")
    env.reset()
    
    K = 5  # would be unknown
    OBS_DIM = 3
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
        
        if i % REFIT_EVERY == REFIT_EVERY - 1:
            create_video(frames, 60, "./video/out")
            try:
                plot_suite(controller, np.stack(obs), np.stack(actions))
                plt.show()
                controller = controller.estimate_and_identify(np.stack(obs), np.stack(actions))
            except Exception as e:
                raise e

    create_video(frames, 60, "./video/out")
    env.close()
