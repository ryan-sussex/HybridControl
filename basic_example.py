import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

from ssm import LDS, SLDS


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=30, nypts=30,
    alpha=0.8, ax=None, figsize=(3, 3)):

    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax



if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    env.action_space.seed(42)

    data = []
    observation, info = env.reset(seed=42)   

    action = 0
    for i in range(1000):
        if i > 500:
            action = 2
        data.append(observation)
        observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    data = np.stack(data)
    pad = np.random.random((data.shape[0], data.shape[1]+ 1))
    pad[:data.shape[0],: data.shape[1]] = data
    data = pad
    print(data.shape)
    lds = LDS(data.shape[1], 1, emissions="gaussian")
    lds.initialize(data)

    y = data
    D_latent = 2   # Latent dimension
    D_obs = data.shape[1]   # Data dimension
    K = 4   # Number of components


    # Fit SLDS
    rslds_lem = SLDS(D_obs, K, D_latent,
                transitions="recurrent_only",
                dynamics="diagonal_gaussian",
                emissions="gaussian_orthog",
                single_subspace=True)

    rslds_lem.initialize(y)
    q_elbos_lem, q_lem = rslds_lem.fit(y, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=1000, alpha=0.0)
    xhat_lem = q_lem.mean_continuous_states[0]
    zhat_lem = rslds_lem.most_likely_states(xhat_lem, y)



    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    lim = abs(xhat_lem).max(axis=0) + 1
    plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
    plt.title("Most Likely Dynamics, Laplace-EM")

    plt.show()