#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:57:24 2024

Need to make sure I have my T.shape stuff the right way around...

@author: pzc
"""

import pymdp
from pymdp import utils
from pymdp.maths import softmax
from pymdp.agent import Agent
from pymdp import learning
from pymdp.maths import spm_log_single
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import entropy
import random
import networkx as nx

def construct_graph():

    # Create a directed graph that will mimic an undirected graph by adding both directional edges
    G = nx.DiGraph()
    # Define edges to add
    edges = [('A', 'B'), ('B', 'C')]

    # Add edges with weights, and ensure each edge is bi-directional
    for u, v in edges:
        G.add_edge(u, v, weight=1)
        G.add_edge(v, u, weight=1)  # Add the reverse edge

    # Adding self-loops for each node
    nodes = ['A', 'B', 'C']
    for node in nodes:
        G.add_edge(node, node, weight=1)  # Adding a self-loop with weight 1

    # Get the adjacency matrix
    Adj = nx.adjacency_matrix(G).toarray()
    
    # Draw the graph using the adjacency matrix A
    pos = nx.circular_layout(G)  # positions for all nodes in a circular layout for better visibility of self-loops and bi-directional edges

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', linewidths=1, font_size=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'))

    # plt.title('Graph Visualization with Self-Loops and Bi-Directional Edges')
    plt.show()
    
    return Adj

def adj_to_transition(Adj):
    
    # convert adjacency matrix to transition matrix
    # normalize each row to sum to 1
    T = np.divide(Adj.T, Adj.sum(axis=1)).T

    # handle case where division by zero occurs (nodes with no outgoing edges)
    T[np.isnan(T)] = 0
    
    return T


def create_A(num_obs, num_states, state_modes, obs_modes):

    """ Identity mapping between observations and states """

    A = utils.initialize_empty_A(num_obs, num_states)
        
    A[0] = np.eye(len(state_modes) )* 1
    
    A[0] = utils.norm_dist(A[0])

    return A

def create_B(adj, mode_action_names, num_states):
    
    """ Take each row and extend for a discrete action """
    
    B = utils.initialize_empty_B(num_states, num_states)
    T = adj_to_transition(adj) 
    
    for i in range(T.shape[1]): # for each row 
        B[0][:, :, i][i] = T[i] * 1/T[i]  # any available paths = T
        B[0][np.isnan(B[0])] = 0
        
    # identify any columns of zeros in each slice
    # get all of the neighbouring nodes to that 0 column node
    # and set the transition probs as uniformly going to neighbouring nodes instead
    for i in range(T.shape[1]):
        for j in range(T.shape[0]): # T.shape[0] should be len of columns
            if np.array_equal(B[0][:,j,i], np.zeros(T.shape[1])):
                B[0][:,j,i] = adj[:,j]/np.sum(adj[:,j], axis=0)
    
    return B
        
    
def create_C(num_obs, rew_idx, pun=-5.,reward=5):
    
    """ create prior preference over mode observations  """
    
    C = utils.obj_array_zeros(num_obs)
    
    C[0] = np.zeros(num_obs[0])
    C[0] + pun
    C[0][rew_idx] = reward
    return C[0]


# %% CONSTRUCT AGENT

# create transition model
adj = construct_graph() # create adjacency matrix

# state
state_modes = [f'{i}' for i in range(adj.shape[0])]
num_states = [len(state_modes)]
num_factors = len(num_states)

# observations
obs_modes = [f'{i}' for i in range(adj.shape[0])]
num_obs = [len(obs_modes)]
num_modalities = len(num_obs)

# actions
mode_action_names = [f'Go-{i}' for i in range(adj.shape[0])]
num_controls = [len(mode_action_names)]

# create observation likelihood
A = create_A(num_obs, num_states, state_modes, obs_modes)

B = create_B(adj, mode_action_names, num_states)

# create prior preferences
rew_idx = 2 # index of the rewarding observation
C = create_C(num_obs, rew_idx, pun=-5,reward=5) 


# construct active inference agent
#my_agent = Agent(A=A, B=B, C=C, use_param_info_gain = False, action_selection='deterministic', alpha=16.0)
my_agent = Agent(A=A, B=B, C=C, policy_len=4, policies=None, B_factor_list=None, action_selection="deterministic")


# In[] Run AIF loop

my_agent.reset() # resets qs and q_pi to uniform
    
init_obs = [1, 0, 0]
qs = my_agent.infer_states(init_obs)
    
q_pi, efe = my_agent.infer_policies()
    
chosen_action_id = my_agent.sample_action()
    
movement_id = int(chosen_action_id[0]) # because context action is always 'do-nothing'
choice_action = mode_action_names[movement_id] # just for recording purposes
print(choice_action)
    
# # step the environment then infer state again so that the dirichlet parameters can properly update
# obs_label = my_env.step(choice_action)
# obs = [reward_obs_names.index(obs_label[0])]
# qs = my_agent.infer_states(obs)
# if learning:
#     qA = my_agent.update_A(obs)
# print(obs_label[0], my_agent.qs[0], choice_action, q_pi)

