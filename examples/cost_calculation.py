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

from hybrid_control.costs import * 
    

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
    
    action = p_0()
    
    traj = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)
        
        # controller logic
        probs = mode_posterior(observation, W, b)   
        idx_mode = np.argmax(probs)
        mode = np.eye(len(probs))[idx_mode]   
        
        if i == 0: # only need do this once
        
            # create cost weighted adjancency matrix  
            cost_matrix = get_cost_matrix(adj, 
                                          priors, 
                                          As, 
                                          Bs,
                                          Q=np.eye(As[0].shape[0])*100,
                                          R=np.eye(Bs[0].shape[1]))
            # alternatively just get random costs
            # random_costs = get_random_cost_matrix(adj) 
        
        # set agents prior over policies to the softmaxed control costs
        agent.E = get_prior_over_policies(agent, cost_matrix, idx_mode, alpha = 0.0001)
        # plt.plot(agent.E)
        # plt.show()
        
        agent, discrete_action = otm.step_active_inf_agent(agent,  mode)
        cts_prior = priors[discrete_action]

        active_linear = env.linear_systems[idx_mode]

        lc = LinearController(
            As[idx_mode],
            Bs[idx_mode],
            Q=np.eye(active_linear.A.shape[0])*100,
            R=np.eye(active_linear.B.shape[1])
        )
        lc = convert_to_servo(lc, cts_prior)

        x_bar = np.r_[observation - cts_prior, 1] # internal coords

        # print(observation)
        action = lc.finite_horizon(x_bar, t=0, T=100)

    print(agent.E)

    prob_hist = [mode_posterior(x, W, b) for x in traj]
    
    
    
    

   
    