import logging
from typing import List
import numpy as np

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo
from hybrid_control.controller import Controller

logging.basicConfig(level=logging.INFO)


def control_prior(discrete_action):
    """
    Takes discrete action, maps to cts point, use lqr to generate cts action
    """
    pass


def p_0():
    return np.random.normal(np.array([0, 0]), 0.1)


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

    controller = Controller(As=As, Bs=Bs, W=W, b=b)

    action = p_0()

    traj = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)

        action = controller.policy(observation)
