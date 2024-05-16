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
    theta = np.arctan(x[1] / x[0])
    x[1] = theta
    return x[1:]


def preprocess(obs: np.ndarray, polar=True):
    if polar:
        obs = to_polar(obs)
    return obs

if __name__ == "__main__":
    POLAR = True
    ENV_STEPS = 2000
    REFIT_EVERY = 500

    env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    env.reset()

    K = 5  # would be unknown
    OBS_DIM = 2 if POLAR else 3
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
        observation = preprocess(observation)
        frames.append(env.render())

        if terminated or truncated:
            observation, info = env.reset()
            observation = preprocess(observation)

        obs.append(observation)
        actions.append(action)

        action = controller.policy(observation, action)

        if i % REFIT_EVERY == REFIT_EVERY - 1:
            create_video(frames, 60, "./video/out")
            try:
                plot_suite(controller, np.stack(obs), np.stack(actions))
                plt.show()
                controller = controller.estimate_and_identify(
                    np.stack(obs), np.stack(actions)
                )
            except Exception as e:
                raise e

    create_video(frames, 60, "./video/out")
    env.close()
