import os 
os.environ['CONFIG_PATH'] = '/home/pzc/Documents/phd/pymdp projects/hybrid-control/config.ini'

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


def to_polar(x: np.ndarray):
    """
    Parameters
    ----------
    x: (state_dim,)
        First two entries should be cartesian coordinates

    Returns
    ---------
    np.ndarray: (state_dim -1, ), first two cordinates replaced with theta
    """
    theta = np.arctan(x[1] / x[0]) - np.pi / 4
    x[1] = theta
    return x[1:]


def preprocess(obs: np.ndarray, polar=True):
    if polar:
        obs = to_polar(obs)
    return obs




if __name__ == "__main__":
    POLAR = False
    REWARD_LOC = np.array([0, 0.]) if POLAR else np.array([0, 1, 0])
    
    ENV_STEPS = 10000
    REFIT_EVERY = 1000

    env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    env.max_episode_steps = 10000
    env.reset()
    max_u = env.action_space.high
    min_u = env.action_space.low

    K = 15 # would be unknown
    K = 6 # would be unknown
    OBS_DIM = 2 if POLAR else 3
    ACT_DIM = 1
    N_ITER = 100
    N_STEPS = 100

    controller = get_initial_controller(OBS_DIM, ACT_DIM, K, max_u=max_u, min_u=min_u)
    # controller.set_known_reward(100, pos=REWARD_LOC)
    action = controller.policy()
    
    accum_reward = 0
    rewards = []

    obs = []
    actions = []
    discrete_actions = []
    frames = []
    for i in range(ENV_STEPS):
            
        observation, reward, terminated, truncated, info = env.step(action)
        observation = preprocess(observation, polar=POLAR)
        frames.append(env.render())
        
        accum_reward += reward
        rewards.append(accum_reward)

        if terminated or truncated:
            observation, info = env.reset()
            observation = preprocess(observation, polar=POLAR)

        obs.append(observation)
        actions.append(action)
        discrete_actions.append(controller.discrete_action)
        
        action = controller.policy(observation, action, reward)

        if i % REFIT_EVERY == REFIT_EVERY - 1:
            create_video(frames, 60, "./video/out")
            observation, info = env.reset()
            try:
                plot_suite(
                    controller,
                    np.stack(obs),
                    np.stack(actions),
                    discrete_actions=discrete_actions,
                    start=i + 1 - REFIT_EVERY,
                    level=2
                )
                plt.show()
                controller = controller.estimate_and_identify(
                    np.stack(obs), np.stack(actions)
                )
            except Exception as e:
                pass

    create_video(frames, 60, "./video/out")
    env.close()

plot_coverage(obs)
plot_total_reward(rewards)