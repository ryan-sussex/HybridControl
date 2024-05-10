#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:52:54 2024

@author: pzc
"""

import logging
import numpy as np
from pymdp.maths import softmax as sm

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo, get_trajectory_cost

def p_0():
    return np.random.normal(np.array([0,0]), .1)


def estimated_system_params(env):
    """
    Warning! Passed env for simulation, real model does not have access 
    """
    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])
    As = [linear.A for linear in env.linear_systems]
    Bs = [linear.B for linear in env.linear_systems]
    return W, b, As, Bs


def get_cost_matrix(adj, priors, As, Bs, Q, R):
    
    costs = adj.copy()
    
    for i in range(adj.shape[1]):
        for j in range(adj.shape[0]):
            if bool(adj[i,j]): # if transition allowed
                x_0 =  priors[j]
                x_ref = priors[i]
                costs[i,j] = get_trajectory_cost(As[j], Bs[j], Q, R, x_0, x_ref)
                
    return costs


def get_random_cost_matrix(adj):
    
    '''
    Temporary function which returns diagonally assymmetric costs with low cost
    for self loops and 0 entries for impossible transitions
    
    '''
    
    # adj2 = adj.copy()
    
    # generate random costs between 1000 and 10000
    random_costs = 1000 + (10000-100) * np.random.rand(adj.shape[0], adj.shape[1])
    
    # make diagonally assymmetric
    random_costs[np.triu_indices(adj.shape[0], 1)] =  0.5 * random_costs[np.tril_indices(adj.shape[0], -1)]
    
    random_costs = adj * random_costs
    
    # make diagonals of random costs 1
    np.fill_diagonal(random_costs, 1)
    
    # make diagonals low cost
    np.fill_diagonal(random_costs, np.diag(random_costs)*1000)
    
    # make negative
    random_costs = random_costs * -1 
    
    return random_costs


def sample_path(B, policy, init_state):
    '''
        Implements MCMC sampling of the transition matrix to give a sequence 
        of states under a policy (sequence of actions)
    '''
    current_state = init_state
    
    trajectory = [current_state]
    
    for action in policy:
        # Determine next state based on current state and action probabilities
        
        probabilities = B[0][:,current_state,action]
        next_state = np.random.choice(range(len(probabilities)), p=probabilities)
        trajectory.append(next_state)
        
        current_state = next_state
        
    return trajectory

def cost_per_path(cost_matrix, path):
    '''
        Calculates the cost of an individual path (sequence of states under a policy) 
        by using the cost_matrix
    '''
    cost = []
    for i in range(len(path)-1):
        # current_state = path[i], next_state = path[i+1]
        # index random_costs by [i, i+1]
        cost.append(random_costs[path[i], path[i+1]])
        
    return cost

def cost_per_policy(B, cost_matrix, policy, init_state):
    '''
        Calculates the total cost of a particular policy by averaging multiple 
        possible paths 
    '''
    # sample a number of paths per policy and take average
    n_samples = 10
    total_cost = []
    for i in range(n_samples):
        # simulate a trajectory
        path = sample_path(B, policy, init_state)
        # get cost of trajectory
        total_cost.append(cost_per_path(cost_matrix, path))
    
    # average over 
    policy_cost = np.sum(total_cost)/n_samples
    
    return policy_cost

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    ENV_STEPS = 10

    env = get_three_region_env(0, 0, 5)

    W, b, As, Bs = estimated_system_params(env)

    priors = generate_all_priors(W, b)
    print(priors)

    adj = extract_adjacency(W, b)
    print(adj)

    agent = otm.construct_agent(adj)
    print(agent.B[0][:, :, 0])
    
    # using adj matrix, all priors, agent.policies and initial mode
    
    # 1. get the initial state of the system
    action = p_0()
    observation, reward, terminated, truncated, info = env.step(action)
    probs = mode_posterior(observation, W, b)
    idx_mode = np.argmax(probs)
    mode = np.eye(len(probs))[idx_mode]
    
    # 2. create cost adj matrix  
    #get closed form solution for cost?
    cost_matrix = get_cost_matrix(adj, 
                                  priors, 
                                  As, 
                                  Bs,
                                  Q=np.eye(As[idx_mode].shape[0])*100,
                                  R=np.eye(Bs[idx_mode].shape[1]))
    
    random_costs = get_random_cost_matrix(adj) # for now just get random costs
  
    # 3. get costs for each policy
    # alpha = 0.001
    alpha = 0.0001
    # get control costs for each policy by indexing i for current state, j for policy action
    pi_costs = np.zeros(len(agent.policies),)
    for i in range(len(agent.policies)):
        policy = np.squeeze(agent.policies[i])
        pi_costs[i] = cost_per_policy(agent.B, cost_matrix=random_costs, policy=policy, init_state=idx_mode)
    
    P_pi = sm(pi_costs*alpha)
    
    # 4. set agent.E to the softmaxed control costs
    agent.E = P_pi
    # sm(p*0.0001) probably need a very low alpha (high temp)
    plt.plot(agent.E)
   
    
    # lift reward to pymdp agent
