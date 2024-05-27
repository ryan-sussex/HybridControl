from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from hybrid_control.controller import Controller
from hybrid_control.observer_transition_model import clean_q_pi

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
    alpha=ALPHA,
    nxpts=30,
    nypts=30,
    ax=None,
    at_state=None,
    at_action=None,
):
    K = controller.n_modes
    D = controller.obs_dim

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    missing = np.zeros((xy.shape[0], D - 2))
    if at_state is not None:
        at_state = at_state.reshape((1, -1))
        missing = np.repeat(at_state, repeats=xy.shape[0], axis=0)
        missing = missing[:, xy.shape[1]:]

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
            alpha=alpha,
        )
    ax.legend([str(i) for i in range(K)])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Most likely dynamics,  dims(0,1)")


    if controller.reward_pos_cts is not None:
        plot_reward(controller, xlim=xlim, ylim=ylim, ax=ax)

    plot_mode_priors(controller, xlim=xlim, ylim=ylim, ax=ax)
    return ax


def plot_actions(
    controller: Controller,
    obs: np.ndarray,
    actions: np.ndarray,
    alpha=ALPHA,
    ax=None,
    xlim=X_LIM,
    ylim=X_LIM,
    discrete_actions: Optional[List[int]] = None,
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

        color = colors[k % len(colors)]
        if discrete_actions is not None:
            color = [colors[i % len(colors)] for i in discrete_actions if i == k]

        ax.quiver(
            obs[zk, 0],
            obs[zk, 1],
            Bu[zk, 0],
            Bu[zk, 1],
            color=color,
            alpha=alpha,
            # label="action taken in state:"
        )
    # ax.legend([f"Go to {i}" for i in range(controller.n_modes)])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Action applied dims(0,1)")
    return ax


def plot_trajectory(
    controller,
    x,
    actions,
    ls="dashed",
    ax=None,
    xlim=X_LIM,
    ylim=X_LIM,
):
    probs = np.stack(
        [controller.mode_posterior(state, action) for state, action in zip(x, actions)]
    )
    z = np.argmax(probs, axis=1)

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)

    ax.plot(x[0, 0], x[0, 1], "go")
    ax.plot(x[-1, 0], x[-1, 1], "r+")

    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(
            x[start : stop + 1, 0],
            x[start : stop + 1, 1],
            lw=1,
            ls=ls,
            color=colors[z[start] % len(colors)],
            alpha=1.0,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Trajectory dims(0,1)")
    return ax


def get_lims(x, scale=0.5):
    xlim = (np.min(x[:, 0]), np.max(x[:, 0]))
    x_diff = xlim[1] - xlim[0]
    ylim = (np.min(x[:, 1]), np.max(x[:, 1]))
    y_diff = ylim[1] - ylim[0]
    xlim = (xlim[0] - scale * x_diff, xlim[1] + scale * x_diff)
    ylim = (ylim[0] - scale * y_diff, ylim[1] + scale * y_diff)
    return xlim, ylim


def get_chunk(obs, actions, discrete_actions, start=0, end=-1):
    obs = obs[start:end]
    actions = actions[start:end]
    if discrete_actions is not None:
        discrete_actions = discrete_actions[start:end]
    return obs, actions, discrete_actions


def plot_overlapped(
    controller: Controller,
    obs: np.ndarray,
    actions: np.ndarray,
    start=0,
    end=-1,
    discrete_actions: Optional[List[int]] = None,
    ax=None,
    xlim=None,
    ylim=None,
):
    obs, actions, discrete_actions = get_chunk(
        obs, actions, discrete_actions, start=start, end=end
    )

    if xlim is None:
        xlim, ylim = get_lims(obs)

    ax = plot_most_likely_dynamics(
        controller, alpha=0.4, xlim=xlim, ylim=ylim, at_state=obs[0], ax=ax
    )
    plot_trajectory(controller, obs, actions, ax=ax, xlim=xlim, ylim=ylim)
    plot_actions(
        controller,
        obs,
        actions,
        ax=ax,
        alpha=1,
        xlim=xlim,
        ylim=ylim,
        discrete_actions=discrete_actions,
    )
    ax.legend()
    return ax


def plot_reward(
    controller: Controller,
    ax=None,
    xlim=None,
    ylim=None,
):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    ax.scatter(controller.reward_pos_cts[0], controller.reward_pos_cts[1], s=200, marker="v", label="Reward", color="black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Trajectory dims(0,1)",)
    return ax

def plot_mode_priors(
    controller: Controller,
    ax=None,
    xlim=None,
    ylim=None,
):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    for i, mode in enumerate(controller.mode_priors):
        ax.scatter(mode[0], mode[1], s=100, marker=">", label=f"goal {i}", )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Trajectory dims(0,1)",)
    return ax
def plot_multiple_overlapped(
    controller: Controller,
    obs: np.ndarray,
    actions: np.ndarray,
    discrete_actions=Optional[List[int]],
):
    _, axs = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True)
    n_plots = len(axs) + 1
    chunksize = int((len(obs) - 1) / n_plots)
    # raise
    xlim, ylim = get_lims(obs)

    start = 0
    end = chunksize
    for i, ax in enumerate(axs):
        plot_overlapped(
            controller,
            obs,
            actions,
            start=start,
            end=end,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            discrete_actions=discrete_actions,
        )
        start += int(chunksize)
        end += int(chunksize)
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
def _plot_efe(efe, q_pi, E, utility=None, state_ig=None, param_ig=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111)
    # plt.plot(efe, label='efe') 
    ax.plot(q_pi, label = 'q_pi')
    ax.plot(E, label='E vector')
    # if utility is not None:
    #     ax.plot(utility, label='util')
    if state_ig is not None:
        ax.plot(state_ig, label='sig')
    if param_ig is not None:
        ax.plot(param_ig, label='pig')
    ax.set_title('Components of EFE')
    ax.legend()


def plot_efe(controller: Controller):
    _, axs = plt.subplots(nrows=1, ncols=controller.n_modes, sharex=True, sharey=True)
    q_pi, efe, utility, state_ig, param_ig = controller.agent.infer_policies_expand_G()
    for idx_mode, ax in enumerate(axs):
        q_pi = clean_q_pi(q_pi, controller.adj, idx_mode, controller.agent)
        _plot_efe(efe, q_pi, controller.agent.E, utility, state_ig, param_ig, ax=ax)
        ax.set_title(f"Components of EFE for mode {idx_mode}")
        
def plot_total_reward(rewards):
    plt.plot(np.linspace(min(rewards), max(rewards), len(rewards)), rewards)
    plt.title('Reward over time')
    plt.show()
    
# def plot_coverage(obs):
#     obs_all = np.squeeze(obs)
#     plt.scatter(obs_all[:,0], obs_all[:,1], s=0.1)
#     # plt.title('state space coverage')
#     plt.xlabel('Position', fontsize=22)
#     plt.ylabel('Velocity', fontsize=22)
#     plt.spines['top'].set_visible(False)
#     plt.spines['right'].set_visible(False)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     plt.tick_params(axis='both', which='minor', labelsize=20)    
#     plt.show()

def plot_coverage(obs):
    obs_all = np.squeeze(obs)
    plt.scatter(obs_all[:, 0], obs_all[:, 1], s=0.1)
    
    plt.xlabel('Position', fontsize=22)
    plt.ylabel('Velocity', fontsize=22)
    plt.xlim(-1.2, 0.6)
    plt.ylim(-0.07, 0.07)
    
    ax = plt.gca()
    
    # Set a fainter grey background color
    ax.set_facecolor('#f0f0f0')  # Lighter shade of grey
    
    # Customize gridlines
    ax.grid(True, which='both', color='white', linewidth=2)
    
    # Adjust the grid z-order to be behind the scatter plot
    ax.set_axisbelow(True)
    
    # Remove the top and right spines (the black box)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set thinner spines for bottom and left
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', labelsize=18)
    
    plt.show()
    
def plot_av_rewards(average_rewards, std_rewards):
    steps = np.arange(len(average_rewards))
    plt.plot(steps, average_rewards, label='Average Reward')
    plt.fill_between(steps, average_rewards - std_rewards, average_rewards + std_rewards, color='b', alpha=0.2)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Average reward per step')
    plt.legend()
    plt.show()
    
    
def plot_av_coverage(obs):
    obs_all = np.squeeze(obs)
    plt.scatter(obs_all[:,0], obs_all[:,1], s=0.1)
    plt.title('Average cumulative state-space coverage')
    plt.show()

def plot_suite(
    controller: Controller,
    obs: np.ndarray,
    actions: np.ndarray,
    discrete_actions: Optional[List[int]] = None,
    rewards: Optional[List[int]] = None,
    start=0,
    end=-1,
    level=3,
):
    obs, actions, discrete_actions = get_chunk(
        obs, actions, discrete_actions, start=start, end=end
    )

    if controller.obs_dim < 2:
        return
    xlim, ylim = get_lims(obs)

    plot_most_likely_dynamics(controller, xlim=xlim, ylim=ylim)

    plot_multiple_overlapped(
        controller, obs, actions, discrete_actions=discrete_actions
    )


    if level <=1:
        return
    plot_overlapped(
        controller, obs, actions, end=100, discrete_actions=discrete_actions
    )

    plot_trajectory(controller, obs, actions, xlim=xlim, ylim=ylim)
    plot_actions(
        controller,
        obs,
        actions,
        xlim=xlim,
        ylim=ylim,
        discrete_actions=discrete_actions,
    )
    # Graphs
    draw_mode_graph(controller)
    draw_cost_graph(controller)

    # EFE
    plot_efe(controller)
    
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






if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # import numpy as np

    # import matplotlib.animation as animation

    # fig, ax = plt.subplots()

    # def f(x, y):
    #     return np.sin(x) + np.cos(y)

    # x = np.linspace(0, 2 * np.pi, 120)
    # y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

    # # ims is a list of lists, each row is a list of artists to draw in the
    # # current frame; here we are just animating one artist, the image, in
    # # each frame
    # ims = []
    # for i in range(60):
    #     x += np.pi / 15
    #     y += np.pi / 30
    #     im = ax.imshow(f(x, y), animated=True)
    #     if i == 0:
    #         ax.imshow(f(x, y))  # show an initial one first
    #     ims.append([im])

    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    #                                 repeat_delay=1000)

    # # To save the animation, use e.g.
    # #
    # # ani.save("movie.mp4")
    # #
    # # or
    # #
    # # writer = animation.FFMpegWriter(
    # #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # # ani.save("movie.mp4", writer=writer)

    # plt.show()
