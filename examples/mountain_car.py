import logging
from typing import List, Callable
import numpy as np
import gym

from ssm import SLDS

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.controller import Controller, get_initial_controller

from hybrid_control.plotting.utils import *

from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


# def gt(env):
#     """
#     Warning! Passed env for simulation, real model does not have access
#     """
#     W_x = np.block([[linear.w] for linear in env.linear_systems])
#     W_u = np.zeros(W_x.shape)
#     b = np.block([linear.b for linear in env.linear_systems])
#     As = [linear.A for linear in env.linear_systems]
#     Bs = [linear.B for linear in env.linear_systems]
#     return W_x, W_u, b, As, Bs


if __name__ == "__main__":
    ENV_STEPS = 200

    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    env.reset()
    
    K = 3  # would be unknown
    OBS_DIM = 2
    ACT_DIM = 1
    N_ITER = 100
    N_STEPS = 100

    controller = get_initial_controller(OBS_DIM, ACT_DIM, K)
    action = controller.policy()

    obs = []
    actions = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        obs.append(observation)
        actions.append(action)

        action = controller.policy(observation, action)

        if i == 100:
            controller = controller.estimate_and_identify(np.stack(obs), np.stack(actions))    
    env.close()


    # # Simple report
    # from hybrid_control.logisitc_reg import mode_posterior
    # W = np.block([[linear.w] for linear in env.linear_systems])
    # b = np.block([linear.b for linear in env.linear_systems])
    # print("Trajectory", obs)
    # print("model", [np.argmax(controller.mode_posterior(x, u)) for x, u in zip(obs, actions)])
    
    # W_x, W_u, b, As, Bs = gt(env)
    # print("gt", [np.argmax(mode_posterior(x, u, W_x, W_u, b)) for x, u in zip(obs, actions)])