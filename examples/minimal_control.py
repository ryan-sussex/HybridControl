import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from hybrid_control.environments.library import get_three_region_env
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


def plot_tensor_heatmap(tensor, axis=2):
    """
    Plots heatmaps for each slice of the given tensor along the specified axis.
    
    Parameters:
    tensor (numpy array): The 3D tensor to be plotted.
    axis (int): The axis along which to slice the tensor. Default is 0.
    """
    # Rearrange the tensor so that the specified axis is the first axis
    tensor = np.moveaxis(tensor, axis, 0)
    
    num_slices = tensor.shape[0]
    
    for i in range(num_slices):
        plt.figure(figsize=(8, 6))
        plt.imshow(tensor[i], cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'action Go-to {i}')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

if __name__ == "__main__":
    # ENV_STEPS = 50
    ENV_STEPS = 50

    env = get_three_region_env(0, 0, 5)

    W, b, As, Bs = estimated_system_params(env)

    # TODO:
    # calculate costs between modes
    # lift reward to pymdp agent

    controller = Controller(As=As, Bs=Bs, bs=None, W_u=None, W_x=W, b=b)
    print("COST", controller.cost_matrix)

    action = p_0(env)

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
