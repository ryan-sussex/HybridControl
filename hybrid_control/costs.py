import logging
import numpy as np
from pymdp.maths import softmax as sm

from hybrid_control.lqr import get_trajectory_cost


def get_cost_matrix(adj, priors, As, Bs, Q, R, bs):
    """
    Parameters
    ----------
    adj : array
        adjacency matrix.
    priors : list of arrays
        control priors for each discrete mode.
    As : list of arrays
        system parameter A matrices.
    Bs : list of arrays
        system parameter B matrices.
    Q : array
        state cost matrix for LQR controller
    R : array
        control cost matrix for LQR controller

    Returns
    -------
    costs_matrix : array
        adjacency matrix weighted by costs of each transition.

    """

    costs_matrix = adj.copy()

    for i in range(costs_matrix.shape[1]):
        for j in range(costs_matrix.shape[0]):
            if bool(adj[i, j]):  # if transition allowed
                x_0 = priors[j]
                x_ref = priors[i]
                costs_matrix[i, j] = get_trajectory_cost(As[j], Bs[j], Q, R, bs[j], x_0, x_ref)

    # make negative
    costs_matrix = costs_matrix * -1
    
    # make impossible costs high
    #costs_matrix[costs_matrix==0] = -100000

    return costs_matrix


def get_random_cost_matrix(adj):
    """
    Temporary function which returns diagonally assymmetric costs with low cost
    for self loops and 0 entries for impossible transitions

    """

    # adj2 = adj.copy()

    # generate random costs between 1000 and 10000
    random_costs = 1000 + (10000 - 100) * np.random.rand(adj.shape[0], adj.shape[1])

    # make diagonally assymmetric
    random_costs[np.triu_indices(adj.shape[0], 1)] = (
        0.5 * random_costs[np.tril_indices(adj.shape[0], -1)]
    )

    random_costs = adj * random_costs

    # make diagonals of random costs 1
    np.fill_diagonal(random_costs, 1)

    # make diagonals low cost
    np.fill_diagonal(random_costs, np.diag(random_costs) * 1000)

    # make negative
    random_costs = random_costs * -1

    return random_costs


def sample_path(B, policy, init_state):
    """
    Parameters
    ----------
    B : array
        Agent's discrete transition matrix.
    policy : array
        Sequence of actions.
    init_state : int
        Initial discrete mode.

    Returns
    -------
    trajectory : array
        Sequence of sampled states given a sequence of actions (policy).

    """
    current_state = init_state

    trajectory = [current_state]

    for action in policy:
        # Determine next state based on current state and action probabilities

        probabilities = B[0][:, current_state, action]
        next_state = np.random.choice(range(len(probabilities)), p=probabilities)
        trajectory.append(next_state)

        current_state = next_state

    return trajectory


def cost_per_path(cost_matrix, path):
    """
    Parameters
    ----------
    cost_matrix : array
        adjacency matrix weighted by control cost.
    path : array
        Sequence of sampled states given a policy.

    Returns
    -------
    cost : float
        cost of an individual path (sequence of states under a policy).

    """
    cost = []
    for i in range(len(path) - 1):
        # current_state = path[i], next_state = path[i+1]
        # index costs by [i, i+1]
        cost.append(cost_matrix[path[i], path[i + 1]])

    cost = sum(cost)

    return cost


def cost_per_policy(B, cost_matrix, policy, init_state):
    """
    Parameters
    ----------
    B : array
        Agent's discrete transition matrix.
    cost_matrix : array
        adjacency matrix weighted by costs of each transition.
    policy : array
       Sequence of actions.
    init_state : int
        Initial discrete mode.

    Returns
    -------
    policy_cost : float
        Total cost of a particular policy (by averaging multiple
        possible paths).

    """
    # sample a number of paths per policy and take average
    n_samples = 10   # TODO: magic number
    costs = []
    for i in range(n_samples):
        # simulate a trajectory
        path = sample_path(B, policy, init_state)
        # get cost of trajectory
        costs.append(cost_per_path(cost_matrix, path))

    # average over
    policy_cost = sum(costs) / n_samples

    return policy_cost


def get_prior_over_policies(agent, cost_matrix, idx_mode, alpha=0.0001):
    """
    Parameters
    ----------
    agent : object
        pymdp agent.
    cost_matrix : array
        adjacency matrix weighted by costs of each transition.
    idx_mode : int
        Initial discrete mode.
    alpha : float
        1/Temperature for softmax function

    Returns
    -------
    P_pi : array
        Probability distribution over policies weighted by their control cost
    """

    # get control costs for each policy by indexing i for current state, j for policy action
    pi_costs = np.zeros(
        len(agent.policies),
    )

    for i in range(len(agent.policies)):
        policy_i = np.squeeze(agent.policies[i])
        pi_costs[i] = cost_per_policy(
            agent.B, cost_matrix, policy=policy_i, init_state=idx_mode
        )

    # softmax with temperature parameter (alpha = 1/T)
    P_pi = sm(pi_costs * alpha)

    return P_pi
