import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from hybrid_control.environments.library import get_2d_spiral, get_skewed_spiral
from hybrid_control.controller import Controller
from hybrid_control.plotting.utils import plot_suite

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


def get_first_bounding_box(As, Bs, u_max):
    A = As[0]
    B = Bs[0]
    max_u = np.array([u_max, u_max])
    x_max = np.linalg.pinv(np.eye(A.shape[0]) - A) @ B @ max_u
    x_min = np.linalg.pinv(np.eye(A.shape[0]) - A) @ B @ (-max_u)
    return x_max, x_min


def extend_modes(As: List, Bs: List, W, b, u_max: float):
    n_states = As[0].shape[0]
    for i in range(4):
        As.append(As[0])
        Bs.append(Bs[0])
    max_u = np.array([u_max, u_max])
    # mixed_u  = np.array([u_max, -u_max])
    W = np.block(
        [[W], [-(np.eye(n_states) - As[0]) - W[0]], [(np.eye(n_states) - As[0]) + W[0]]]
    )
    print(W.shape)
    # bs = np.block([b.T, -Bs[0]@max_u])
    bs = np.block([b.T, -Bs[0] @ max_u, -Bs[0] @ max_u])
    print(bs.shape)
    return As, Bs, W, bs


if __name__ == "__main__":
    ENV_STEPS = 500
    REWARD_POS = np.array([1, 1]) * 4
    U_MAX = 1

    env = get_2d_spiral(u_max=U_MAX)
    # env = get_skewed_spiral(u_max=U_MAX)

    W, b, As, Bs = estimated_system_params(env)

    # As, Bs, W, b = extend_modes(As, Bs, W, b, u_max=U_MAX)

    print(len(As))

    controller = Controller(
        As=As, Bs=Bs, bs=None, W_u=None, W_x=W, b=b, reward_pos_cts=REWARD_POS
    )

    # print(
    #     "minimum time:", controller.final_controller.minimum_time(x_0=np.array([0, 0]))
    # )

    print("COST", controller.cost_matrix)

    action = controller.policy()

    obs = []
    actions = []
    discrete_actions = []
    for i in range(ENV_STEPS):

        # plot_tensor_heatmap(controller.agent.B[0])
        observation, reward, terminated, truncated, info = env.step(action)

        obs.append(observation)
        actions.append(action)
        discrete_actions.append(controller.discrete_action)

        action = controller.policy(observation, action)

    plot_suite(
        controller,
        np.stack(obs),
        np.stack(actions),
        discrete_actions=discrete_actions,
    )
    plt.show()

    # Simple report
    from hybrid_control.logisitc_reg import mode_posterior

    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])
    print("Trajectory", obs)
    print("model", [np.argmax(controller.mode_posterior(x)) for x in obs])
    # print("gt", [np.argmax(mode_posterior(x, u, W_x, W_u, b)) for x, u in zip(obs, actions)])
