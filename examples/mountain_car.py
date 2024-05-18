import logging
from typing import List, Callable
import numpy as np
import gym
import matplotlib.pyplot as plt

from ssm import SLDS

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.controller import Controller, get_initial_controller

from hybrid_control.plotting.utils import *

from utils import create_video

logging.basicConfig(level=logging.INFO)


REWARD_LOC = np.array([.5, 5.])

if __name__ == "__main__":
    ENV_STEPS = 10000
    REFIT_EVERY = 1000

    env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env.reset()
    max_u = env.action_space.high
    min_u = env.action_space.low  
    K = 5  # would be unknown
    OBS_DIM = 2
    ACT_DIM = 1
    N_ITER = 100
    N_STEPS = 100

    controller = get_initial_controller(OBS_DIM, ACT_DIM, K, max_u=max_u, min_u=min_u)
    # controller.set_known_reward(100, pos=REWARD_LOC)
    action = controller.policy()

    obs = []
    actions = []
    discrete_actions = []
    frames = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            if terminated > 0:
                action = controller.policy(observation, action, reward) # Here to make sure reward loc is updated
            observation, info = env.reset()

        obs.append(observation)
        actions.append(action)
        discrete_actions.append(controller.discrete_action)

        action = controller.policy(observation, action)
        
        if i % REFIT_EVERY == REFIT_EVERY - 1:
            create_video(frames, 60, "./video/out")
            try:
                # plot_suite(
                #     controller,
                #     np.stack(obs),
                #     np.stack(actions),
                #     discrete_actions=discrete_actions,
                #     start=i + 1 - REFIT_EVERY,
                # )
                # plt.show()
                controller = controller.estimate_and_identify(
                    np.stack(obs), np.stack(actions)
                )
            except Exception as e:
                pass

    create_video(frames, 60, "./video/out")
    env.close()
