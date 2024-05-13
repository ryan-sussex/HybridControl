import logging
from typing import List
import numpy as np

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.controller import Controller

logging.basicConfig(level=logging.INFO)


def p_0(env):
    obs_dim = env.linear_systems[0].A.shape[0]
    return np.random.normal(np.zeros(obs_dim), 0.1)


def estimated_system_params(env):
    """
    Warning! Passed env for simulation, real model does not have access
    """
    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])
    As = [linear.A for linear in env.linear_systems]
    Bs = [linear.B for linear in env.linear_systems]
    return W, b, As, Bs


if __name__ == "__main__":
    ENV_STEPS = 10

    env = get_three_region_env(0, 0, 5)

    W, b, As, Bs = estimated_system_params(env)

    # TODO:
    # calculate costs between modes
    # lift reward to pymdp agent

    controller = Controller(As=As, Bs=Bs, bs=None, W_u=None, W_x=W, b=b)
    print("COST", controller.cost_matrix)

    action = p_0(env)

    obs = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        obs.append(observation)

        action = controller.policy(observation, action)


    # Simple report
    from hybrid_control.logisitc_reg import mode_posterior
    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])
    print("Trajectory", obs)
    print("model", [np.argmax(controller.mode_posterior(x)) for x in obs])
    # print("gt", [np.argmax(mode_posterior(x, u, W_x, W_u, b)) for x, u in zip(obs, actions)])
