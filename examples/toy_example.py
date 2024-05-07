from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ssm import LDS, SLDS
from library import get_linearly_seperated_env
from utils import *


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
            obs = data_to_array(data, actions)
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

    rslds.initialize(obs)

    q_elbos, q = rslds.fit(
        obs,
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


if __name__ == "__main__":
    # env = gym.make("MountainCar-v0", render_mode="human")
    # env.action_space.seed(42)
    K = 2
    N_ITER = 100
    N_STEPS = 100
    env = get_linearly_seperated_env()


    results = system_identification(
        env, 
        k_components=2, 
        env_steps=N_STEPS, 
        varitional_iter=N_ITER
    )



    xhat = results["q"].mean_continuous_states[0]
    zhat = results["rslds"].most_likely_states(xhat, results["obs"])

    Ws, Rs, rs = results["rslds"].transitions.params
    for i in range(len(Ws)):
        print("w", Rs[i])
        print("b", rs[i])

    true_hplane = env.linear_systems[0]
    Rs = np.vstack([Rs, env.linear_systems[0].w])
    rs = np.hstack([rs, env.linear_systems[0].b])
    Rs = np.vstack([Rs, env.linear_systems[1].w])
    rs = np.hstack([rs, env.linear_systems[1].b])

    plot_phases(Rs, rs)

    lim = abs(xhat).max(axis=0) + 1
    plot_most_likely_dynamics(
        results["rslds"], xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=None
    )

    plt.show()#



    # plt.figure(figsize=(6, 6))
    # ax4 = plt.subplot(131)
    # plot_trajectory(zhat, xhat, ax=ax4)
    # plt.title("Inferred, Laplace-EM")

    # Plots
    # original
    # plt.figure(figsize=(6, 6))
    # ax1 = plt.subplot(131)
    # #position
    # plot_original(results["obs"], ax=ax1)