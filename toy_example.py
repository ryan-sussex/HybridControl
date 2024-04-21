from typing import List
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from environments import env, hyperplane

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

from ssm import LDS, SLDS


def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(
            x[start : stop + 1, 0],
            x[start : stop + 1, 1],
            lw=1,
            ls=ls,
            color=colors[z[start] % len(colors)],
            alpha=1.0,
        )

    return ax


def plot_original(x, ax=None, ls="-"):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    ax.plot(
        x[:, 0],
        x[:, 1],
        lw=1,
        ls=ls,
        # color=colors[z[start] % len(colors)],
        alpha=1.0,
    )

    return ax

def plot_most_likely_dynamics(
    model,
    xlim=(-4, 4),
    ylim=(-3, 3),
    nxpts=30,
    nypts=30,
    alpha=0.8,
    ax=None,
    figsize=(3, 3),
):

    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None
    )
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(
                xy[zk, 0],
                xy[zk, 1],
                dxydt_m[zk, 0],
                dxydt_m[zk, 1],
                color=colors[k % len(colors)],
                alpha=alpha,
            )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    plt.tight_layout()

    return ax


def plot_phases(Ws, rs, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

    min_ = min(rs)
    max_ = max(rs)
    x = np.linspace(min_, max_, 100)
    for i in range(len(Ws)-1):
        y = rs[i] - Ws[i][0]/Ws[i][1] * x
        ax.plot(x, y)
    y = rs[-1] - Ws[-1][0]/Ws[-1][1] * x
    ax.plot(x, y, linestyle="dotted")



def add_derivatives(data):
    v = np.diff(data, axis=0)
    # a = np.diff(data, axis=0, n=2)
    return np.hstack((data[: v.shape[0], :], v[: v.shape[0], :]))


def policy(env, obs, t, horizon=200):
    return np.random.normal(np.array([0,0]), 1)


def p_0():
    return np.random.normal(np.array([0,0]), .1)


def data_to_array(data: List, actions: List):
    # convert to arrays and add derivatives
    data = np.stack(data)
    actions = np.stack(actions)
    # data = add_derivatives(data)
    # data = np.hstack((data, actions[:data.shape[0]]))
    return data

if __name__ == "__main__":
    # env = gym.make("MountainCar-v0", render_mode="human")
    # env.action_space.seed(42)
    STEPS = 1000
    ITERS = 100

    data = []
    actions = []
    observation, info = env.reset(seed=42)

    action = p_0()

    for i in tqdm(range(STEPS)):

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


    data = obs
    y = data
    D_obs = data.shape[1]  # Data dimension
    # D_latent = 4  # Latent dimension
    D_latent = D_obs  # Latent dimension

    K = 2  # Number of components

    # Fit SLDS
    rslds = SLDS(
        D_obs,
        K,
        D_latent,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian_id",
        single_subspace=True,
    )

    rslds.initialize(y)
    q_elbos, q = rslds.fit(
        y,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        initialize=False,
        num_iters=ITERS,
        alpha=0.0,
    )
    xhat = q.mean_continuous_states[0]
    zhat = rslds.most_likely_states(xhat, y)

    Ws, Rs, rs = rslds.transitions.params
    for i in range(len(Ws)):
        print("w", Rs[i])
        print("b", rs[i])

    Rs = np.vstack([Rs, hyperplane.c])
    rs = np.hstack([rs, hyperplane.d])

    
    # def get_region()


    
    # plt.figure(figsize=(6, 6))
    # ax1 = plt.subplot(131)
    plot_phases(Rs, rs)



    # Plots
    # original
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(131)
    #position
    plot_original(data, ax=ax1)
    # vel
    # plt.figure(figsize=(6, 6))
    # ax2 = plt.subplot(131)
    # plot_original(data[:, 2:], ax=ax2)
    # # acc
    # plt.figure(figsize=(6, 6))
    # ax3 = plt.subplot(131)
    # plot_original(data[:, 2:], ax=ax3)


    plt.figure(figsize=(6, 6))
    ax4 = plt.subplot(131)
    plot_trajectory(zhat, xhat, ax=ax4)
    plt.title("Inferred, Laplace-EM")

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    lim = abs(xhat).max(axis=0) + 1
    plot_most_likely_dynamics(
        rslds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax
    )
    plt.title("Most Likely Dynamics, Laplace-EM")
    plt.show()
