import logging
from typing import List
import numpy as np
from pymdp.maths import softmax as sm
import math


from hybrid_control.lqr import LinearController


def get_cost_matrix(adj, priors, controllers: List[List[LinearController]]):
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

    for i in range(costs_matrix.shape[0]):
        for j in range(costs_matrix.shape[1]):
            if bool(adj[i, j]):  # if transition allowed
                x_0 = priors[i]
                costs_matrix[i, j] = controllers[i][j].get_trajectory_cost(x_0)

    # make negative
    costs_matrix = costs_matrix * -1
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


def get_prior_over_policies(adj, agent, cost_matrix, idx_mode):
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
        
    # # get appropriate alpha value
    largest_value = np.min(pi_costs) * -1
    order_mag = math.floor(math.log(largest_value, 10))
    alpha = alpha = 10 ** -order_mag

    # softmax with temperature parameter (alpha = 1/T)
    P_pi = sm(pi_costs * alpha)
        
    # find disallowed sequences from adj
    null_seqs = np.argwhere(adj == 0)
    null_seqs = null_seqs.tolist()
    p = np.squeeze(agent.policies)

    # init bool mask 
    mask = np.zeros(p.shape[0], dtype=bool)
    
    # Check for disallowed sequences in policy and update the mask
    for seq in null_seqs:
        mask |= np.any((p[:, :-1] == seq[0]) & (p[:, 1:] == seq[1]), axis=1)
        
    # set prob of policies containing disallowed transitions to zero
    P_pi[mask] = 0
    
    # normalise P_pi leaving zeros as zeros
    non_zero_sum = np.sum(P_pi[P_pi != 0])
    P_pi_norm = P_pi.copy()
    P_pi_norm[P_pi != 0] /= non_zero_sum
    
    return P_pi_norm
