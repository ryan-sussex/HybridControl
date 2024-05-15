from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from hybrid_control.controller import Controller


color_names = ["windows blue", "red", "amber", "faded green", "purple", "grey"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

FIGSIZE = (6, 6)
ALPHA = 0.8
X_LIM = (-2, 2)


def plot_most_likely_dynamics(
    controller: Controller,
    xlim=X_LIM,
    ylim=X_LIM,
    nxpts=30,
    nypts=30,
    ax=None,
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
        # if zk.sum(0) > 0:
        ax.quiver(
            xy[zk, 0],
            xy[zk, 1],
            dxydt_m[zk, 0],
            dxydt_m[zk, 1],
            color=colors[k % len(colors)],
            alpha=ALPHA,
        )
    ax.legend([str(i) for i in range(K)])
    ax.set_xlim(X_LIM)
    ax.set_ylim(X_LIM)
    ax.set_title("Most likely dynamics,  dims(0,1)")
    return


def plot_actions(
    controller: Controller, obs: np.ndarray, actions: np.ndarray, alpha=0.5, ax=None
):
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
        # if zk.sum(0) > 0:
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
    ls="dashed",
    ax=None,
):
    probs = np.stack(
        [controller.mode_posterior(state, action) for state, action in zip(x, actions)]
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


def draw_graph(adj, ax=None):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    G = nx.from_numpy_array(adj)

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=500,
        edge_color="k",
        linewidths=1,
        font_size=15,
    )
    labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='r', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    return


def draw_mode_graph(controller: Controller):
    draw_graph(controller.adj)
    return


def draw_cost_graph(controller: Controller):
    draw_graph(controller.cost_matrix)
    return

    # Show plot


def plot_suite(controller: Controller, obs: np.ndarray, actions: np.ndarray):
    if controller.obs_dim < 2:
        return
    plot_most_likely_dynamics(controller)
    plot_trajectory(controller, obs, actions)
    plot_actions(controller, obs, actions)
    draw_mode_graph(controller)
    draw_cost_graph(controller)
    return


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
