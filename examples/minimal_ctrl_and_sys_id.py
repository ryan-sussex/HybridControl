import logging
from typing import List, Callable
import numpy as np

from ssm import SLDS

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.controller import Controller

from hybrid_control.plotting.utils import *

from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def system_identification(
        env,
        k_components: int,
        env_steps: int = 1000,
        varitional_iter: int = 100,
        initial_policy: Callable = p_0
):
    data = []
    actions = []
    observation, info = env.reset(seed=42)

    action = initial_policy()

    for i in tqdm(range(env_steps)):
        observation, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     observation, info = env.reset() 
        data.append(observation)
        actions.append(action)
        if i > 5:
            obs, acts = data_to_array(data, actions)
            # action += np.random.normal(loc=0, scale=1)
            action = policy(env, obs, t=i)
   
        if observation.dot(observation) > 50:
            env.reset()

    env.close()


    D_obs = obs.shape[1]  # Data dimension
    D_latent = D_obs  # Latent dimension
    # Fit SLDS
    rslds = SLDS(
        D_obs,
        k_components,
        D_latent,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian_id",
        single_subspace=True,
    )

    rslds.initialize(obs, inputs=acts)

    q_elbos, q = rslds.fit(
        obs,
        inputs=acts,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        initialize=False,
        num_iters=varitional_iter,
        alpha=0.0,
    )

    results = dict(
        rslds=rslds,
        obs=obs,
        q_elbos=q_elbos,
        q=q
    )
    return results




def p_0(env):
    obs_dim = env.linear_systems[0].A.shape[0]
    return np.random.normal(np.zeros(obs_dim), 0.1)


def estimated_system_params(rslds: SLDS, env):
    """
    Warning! Passed env for simulation, real model does not have access
    """
    dynamic_params = rslds.dynamics.params
    emission_params = rslds.emissions.params
    softmax_params = rslds.transitions.params

    _, W, b = softmax_params
    As, bs, Bs, Sigmas = dynamic_params

    # W = np.block([[linear.w] for linear in env.linear_systems])
    # b = np.block([linear.b for linear in env.linear_systems])
    # As = [linear.A for linear in env.linear_systems]
    Bs = [linear.B for linear in env.linear_systems]
    return W, b, As, Bs


if __name__ == "__main__":
    ENV_STEPS = 10

    env = get_three_region_env(0, 0, 5)
    # env = get_three_region_env()
    
    K = len(env.linear_systems)  # would be unknown
    N_ITER = 100
    N_STEPS = 100

    results = system_identification(
        env, 
        k_components=K, 
        env_steps=N_STEPS, 
        varitional_iter=N_ITER
    )

    W, b, As, Bs = estimated_system_params(results["rslds"], env)

    # TODO:
    # calculate costs between modes
    # lift reward to pymdp agent

    controller = Controller(As=As, Bs=Bs, W=W, b=b)

    action = p_0(env)

    traj = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)

        action = controller.policy(observation)
