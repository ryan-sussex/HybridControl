#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:57:24 2024

@author: pzc
"""
import logging
from typing import Optional

from pymdp import utils
from pymdp.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


from hybrid_control.config import param_config
from hybrid_control.agent import Agent


logger = logging.getLogger("discrete_controller")



REWARD_MAG = param_config.getint("REWARD_MAG", 5)
POLICY_LEN = param_config.getint("POLICY_LEN", 3)



def construct_graph():

    # Create a directed graph that will mimic an undirected graph by adding both directional edges
    G = nx.DiGraph()
    # Define edges to add
    edges = [("A", "B"), ("B", "C")]

    # Add edges with weights, and ensure each edge is bi-directional
    for u, v in edges:
        G.add_edge(u, v, weight=1)
        G.add_edge(v, u, weight=1)  # Add the reverse edge

    # Adding self-loops for each node
    nodes = ["A", "B", "C"]
    for node in nodes:
        G.add_edge(node, node, weight=1)  # Adding a self-loop with weight 1

    # Get the adjacency matrix
    Adj = nx.adjacency_matrix(G).toarray()

    # Draw the graph using the adjacency matrix A
    pos = nx.circular_layout(
        G
    )  # positions for all nodes in a circular layout for better visibility of self-loops and bi-directional edges

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="k",
        linewidths=1,
        font_size=15,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, "weight")
    )

    # plt.title('Graph Visualization with Self-Loops and Bi-Directional Edges')
    plt.show()

    return Adj


def draw_graph(adj):

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on adjacency matrix
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] == 1:
                G.add_edge(i, j)

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        edge_color="k",
        linewidths=1,
        font_size=15,
    )

    # Show plot
    plt.show()


def adj_to_transition(Adj):

    # convert adjacency matrix to transition matrix
    # normalize each row to sum to 1
    T = np.divide(Adj.T, Adj.sum(axis=1)).T

    # handle case where division by zero occurs (nodes with no outgoing edges)
    T[np.isnan(T)] = 0

    return T


def create_A(num_obs, num_states, state_modes, obs_modes):
    """Identity mapping between observations and states"""

    A = utils.initialize_empty_A(num_obs, num_states)

    A[0] = np.eye(len(state_modes)) * 1

    A[0] = utils.norm_dist(A[0])

    return A


def create_B(adj, mode_action_names, num_states):
    """Take each row and extend for a discrete action"""
    prob = 0.8  # confidence in allowed transitions
    B = utils.initialize_empty_B(num_states, num_states)
    T = adj_to_transition(adj)

    # for each slice of the matrix
    for i in range(B[0].shape[2]):
        # add the relevant row of the adjacency matrix
        B[0][:, :, i][i] += prob * adj[i]
        # for each row of the slice
        for j in range(B[0][:, :, i].shape[0]):
            # make the zero values instead equal to (1-prob)
            B[0][:, j, i][B[0][:, j, i] == 0] = (1 - prob) / B[0][:, j, i][
                B[0][:, j, i] == 0
            ].shape[0]
        # make sure disallowed transitions are set back zero
        B[0][:, :, i][i] *= adj[i]
        # make sure each column is normalised to sum to 1
        for k in range(B[0][:, :, i].shape[1]):
            B[0][:, k, i] = B[0][:, k, i] / np.sum(B[0][:, k, i])

    return B


def create_C(num_obs, rew_idx, pun=0, reward=5):
    """create prior preference over mode observations"""
    if rew_idx == None:
        C = utils.obj_array_zeros(num_obs)
        C[0] += 1
    else:
        C = utils.obj_array_zeros(num_obs)
        C[0][:] = pun
        C[0][rew_idx] = reward
    return C


def construct_agent(adj: np.ndarray, rwd_idx: Optional[int]) -> Agent:
    # state
    state_modes = [f"{i}" for i in range(adj.shape[0])]
    num_states = [len(state_modes)]
    num_factors = len(num_states)

    # observations
    obs_modes = [f"{i}" for i in range(adj.shape[0])]
    num_obs = [len(obs_modes)]
    num_modalities = len(num_obs)

    # actions
    mode_action_names = [f"Go-{i}" for i in range(adj.shape[0])]
    num_controls = [len(mode_action_names)]

    # create observation likelihood
    A = create_A(num_obs, num_states, state_modes, obs_modes)
    B = create_B(adj, mode_action_names, num_states)
    pB = utils.dirichlet_like(B,scale=1)
    pB.any()[pB.any()==0.5] = 1e10 # make impossible transitions unlearnable 
    # create prior preferences

    C = create_C(num_obs, rwd_idx, pun=-REWARD_MAG, reward=REWARD_MAG)

    agent = Agent(
        A=A,
        B=B,
        pB=pB,
        C=C,
        policy_len=POLICY_LEN,
        policies=None,
        B_factor_list=None,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=True,
        action_selection="deterministic",
        lr_pB=10,
    )

    agent.mode_action_names = mode_action_names

    return agent

def plot_efe(efe, q_pi, E, utility=None, state_ig=None, param_ig=None):
    
    # plt.plot(efe, label='efe') 
    plt.plot(q_pi, label = 'q_pi')
    plt.plot(E, label='E vector')
    # if utility is not None:
    #     plt.plot(utility, label='util')
    if state_ig is not None:
        plt.plot(state_ig, label='sig')
    if param_ig is not None:
        plt.plot(param_ig, label='pig')
    plt.title('Components of EFE')
    plt.legend()
    plt.show()
    
def clean_q_pi(q_pi, adj, idx_mode, agent):
    
    '''
    Sets disallowed transitions to zero (considers also disallowes transitions
                                         given current inferred mode)
    '''
    
    # find disallowed sequences from adj
    null_seqs = np.argwhere(adj == 0)
    null_seqs = null_seqs.tolist()
    
    # push current state into first column of policies
    p = np.squeeze(agent.policies)
    new_col = np.ones(p.shape[0])*idx_mode
    new_col = new_col.reshape(-1, 1)
    p = np.hstack((new_col, p))

    # init bool mask 
    mask = np.zeros(p.shape[0], dtype=bool)
    
    # check for disallowed sequences in policy and update mask
    for seq in null_seqs:
        mask |= np.any((p[:, :-1] == seq[0]) & (p[:, 1:] == seq[1]), axis=1)
        
    # set prob of policies containing disallowed transitions to zero
    q_pi[mask] = 0
    
    # normalise P_pi leaving zeros as zeros
    non_zero_sum = np.sum(q_pi[q_pi != 0])
    q_pi_norm = q_pi.copy()
    q_pi_norm[q_pi != 0] /= non_zero_sum
    
    return q_pi_norm


def step_active_inf_agent(adj, idx_mode, agent: Agent, obs: np.ndarray):
    agent.reset()  # resets qs and q_pi to uniform

    qs = agent.infer_states(obs, distr_obs=True)

    if agent.qs_prev is not None:
        agent.qB = agent.update_B(agent.qs_prev)
    
    # NOTE: if plotting different components contributing to EFE, this only works 
    # with a modified infer_policies_expand_G()
    q_pi, efe, utility, state_ig, param_ig = agent.infer_policies_expand_G()

    agent.q_pi = clean_q_pi(q_pi, adj, idx_mode, agent)
       
    # plot_efe(efe, q_pi, agent.E, utility, state_ig, param_ig)
    try:
        chosen_action_id = agent.sample_action()
    except Exception as e:
        logger.warn(e, exc_info=True)
        chosen_action_id = 0    # TODO: figure out why Nans appear

    movement_id = int(
        chosen_action_id[0]
    )  # because context action is always 'do-nothing'
    choice_action = agent.mode_action_names[movement_id]  # just for recording purposes
    logger.info(f"chose action:{choice_action}")

    agent.qs_prev = qs  # for updating pB on next loop

    return agent, movement_id


if __name__ == "__main__":

    from hybrid_control.algebra import *

    # %% CONSTRUCT AGENT

    # create transition model
    # adj = construct_graph() # create adjacency matrix

    W = np.random.randint(-10, 10, (3, 2))
    b = np.random.randint(-10, 10, (3, 1))
    adj = extract_adjacency(W, b)
    draw_graph(adj)

    # state
    state_modes = [f"{i}" for i in range(adj.shape[0])]
    num_states = [len(state_modes)]
    num_factors = len(num_states)

    # observations
    obs_modes = [f"{i}" for i in range(adj.shape[0])]
    num_obs = [len(obs_modes)]
    num_modalities = len(num_obs)

    # actions
    mode_action_names = [f"Go-{i}" for i in range(adj.shape[0])]
    num_controls = [len(mode_action_names)]

    # create observation likelihood
    A = create_A(num_obs, num_states, state_modes, obs_modes)

    B = create_B(adj, mode_action_names, num_states)
    pB = utils.dirichlet_like(B, scale=1)
    # temp = B[0]

    # create prior preferences
    rew_idx = 2  # index of the rewarding observation
    C = create_C(num_obs, rew_idx, pun=-5, reward=5)

    # construct active inference agent
    my_agent = Agent(
        A=A,
        B=B,
        pB=pB,
        C=C,
        policy_len=4,
        policies=None,
        B_factor_list=None,
        action_selection="deterministic",
    )

    # In[] RUN ACTIVE INFERENCE LOOP
    my_agent.reset()  # resets qs and q_pi to uniform

    init_obs = [1, 0, 0]
    qs = my_agent.infer_states(init_obs)

    q_pi, efe = my_agent.infer_policies()

    chosen_action_id = my_agent.sample_action()

    movement_id = int(
        chosen_action_id[0]
    )  # because context action is always 'do-nothing'
    choice_action = mode_action_names[movement_id]  # just for recording purposes
    print(choice_action)
