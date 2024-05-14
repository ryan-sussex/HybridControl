from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from hybrid_control.controller import Controller


color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

FIGSIZE=(6,6)
ALPHA = .8
X_LIM = (-2 , 2)


# def plot_trajectory(z, x, ax=None, ls="-"):
#     zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
#     if ax is None:
#         fig = plt.figure(figsize=(4, 4))
#         ax = fig.gca()
#     for start, stop in zip(zcps[:-1], zcps[1:]):
#         ax.plot(
#             x[start : stop + 1, 0],
#             x[start : stop + 1, 1],
#             lw=1,
#             ls=ls,
#             color=colors[z[start] % len(colors)],
#             alpha=1.0,
#         )

#     return ax


# def plot_original(x, ax=None, ls="-"):
#     if ax is None:
#         fig = plt.figure(figsize=(4, 4))
#         ax = fig.gca()
#     ax.plot(
#         x[:, 0],
#         x[:, 1],
#         lw=1,
#         ls=ls,
#         # color=colors[z[start] % len(colors)],
#         alpha=1.0,
#     )

#     return ax


def plot_most_likely_dynamics(
    controller: Controller,
    xlim=X_LIM,
    ylim=X_LIM,
    nxpts=30,
    nypts=30,
    alpha=ALPHA,
    ax=None,
    figsize=(6, 6),
):
    K = controller.n_modes
    D = controller.obs_dim

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    missing = np.zeros((xy.shape[0], D - 2))

    states = np.c_[xy, missing]
    actions = np.zeros((xy.shape[0], controller.action_dim))

    probs = np.stack(
        [
            controller.mode_posterior(state, action)
            for state, action in zip(states, actions)
        ]
    )
    z = np.argmax(probs, axis=1)

    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(controller.As, controller.bs)):
        dxydt_m = states.dot(A.T) + b - states

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
            ax.legend([str(i) for i in range(K)])
            ax.set_xlim(X_LIM)
            ax.set_ylim(X_LIM)
            ax.set_title("Most likely dynamics,  dims(0,1)")
    return


def plot_actions(controller: Controller, obs: np.ndarray, actions: np.ndarray, alpha=.5, ax=None):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    
    
    probs = np.stack(
        [
            controller.mode_posterior(state, action)
            for state, action in zip(obs, actions)
        ]
    )
    z = np.argmax(probs, axis=1)
    for k, B in enumerate(controller.Bs):
        Bu = actions @ B.T + obs

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(
                obs[zk, 0],
                obs[zk, 1],
                Bu[zk, 0],
                Bu[zk, 1],
                color=colors[k % len(colors)],
                alpha=alpha,
            )
            ax.legend([str(i) for i in range(controller.n_modes)])
            ax.set_xlim(X_LIM)
            ax.set_ylim(X_LIM)
            ax.set_title("Action applied dims(0,1)")
    return


def plot_trajectory(
        controller,
        x,
        actions,
        ls="-",
        ax=None,
):
    probs = np.stack(
        [
            controller.mode_posterior(state, action)
            for state, action in zip(x, actions)
        ]
    )
    z = np.argmax(probs, axis=1)
    
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(
            x[start : stop + 1, 0],
            x[start : stop + 1, 1],
            lw=1,
            ls=ls,
            color=colors[z[start] % len(colors)],
            alpha=1.0,
        )
        ax.set_xlim(X_LIM)
        ax.set_ylim(X_LIM)
        ax.set_title("Trajectory dims(0,1)")
    return



def plot_suite(controller: Controller, obs: np.ndarray, actions:np.ndarray):
    plot_most_likely_dynamics(controller)
    plot_trajectory(controller, obs, actions)
    plot_actions(controller, obs, actions)
    return 



def _plot_most_likely_dynamics(
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


def plot_phases(Ws, rs, ax=None, linestyle=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    min_ = -10
    max_ = 10
    x = np.linspace(min_, max_, 100)
    for i in range(len(Ws)):
        y = rs[i] - Ws[i][0] / Ws[i][1] * x
        ax.plot(x, y, linestyle=linestyle)


def add_derivatives(data):
    v = np.diff(data, axis=0)
    # a = np.diff(data, axis=0, n=2)
    return np.hstack((data[: v.shape[0], :], v[: v.shape[0], :]))


def policy(env, obs, t, horizon=200):
    return np.random.normal(np.array([0, 0]), 1)


def p_0():
    return np.random.normal(np.array([0, 0]), 0.1)


def data_to_array(data: List, actions: List):
    # convert to arrays and add derivatives
    data = np.stack(data)
    actions = np.stack(actions)
    # data = add_derivatives(data)
    # data = np.hstack((data, actions[:data.shape[0]]))
    return data, actions
