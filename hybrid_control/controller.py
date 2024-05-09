import logging
from typing import List
import numpy as np

from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo


logging.basicConfig(level=logging.INFO)



class Controller:

    def __init__(
            self, 
            As: List[np.ndarray], 
            Bs: List[np.ndarray], 
            W: np.ndarray, 
            b: np.ndarray
    ):
        self.n_modes = len(As)
        self.mode_priors = generate_all_priors(W, b)
        self.agent = get_discrete_controller(W, b)
        self.cts_ctrs = get_all_cts_controllers(As, Bs, self.mode_priors)
        self.W = W
        self.b = b

    def mode_posterior(self, observation):
        return mode_posterior(observation, self.W, self.b)


    def policy(self, observation):
        """
        Takes a continuous observation, outputs continuous action.
        """
        probs = self.mode_posterior(observation)
        idx_mode = np.argmax(probs)
        mode = np.eye(len(probs))[idx_mode]  # one-hot rep
        # Discrete 
        self.agent, discrete_action = otm\
            .step_active_inf_agent(self.agent, mode)
        cts_prior = self.mode_priors[discrete_action]
        # Continuous
        cts_ctr = self.cts_ctrs[discrete_action][idx_mode]
        x_bar = np.r_[observation - cts_prior, 1]  # internal coords TODO: simplify this
        action = cts_ctr.finite_horizon(x_bar, t=0, T=100)  # TODO: magic numbers
        return action


def get_discrete_controller(W, b):
    adj = extract_adjacency(W, b)
    return otm.construct_agent(adj)


def get_cts_controller(As, Bs, i: int, j: int, mode_priors: List):
    """
    Constructs the controller for traversing region i to reach goal j
    """
    lc = LinearController(
        As[i],
        Bs[i],
        Q=np.eye(As[i].shape[0]) * 100,   # TODO: Magic numbers
        R=np.eye(Bs[i].shape[0]),
    )
    return convert_to_servo(lc, mode_priors[j])


def get_all_cts_controllers(As, Bs, mode_priors: List):
    """
    Returns list of lists, where element list[i][j] is the controller
    for going from region i, to the prior specified by mode_prior[j]
    """
    n_modes = len(mode_priors)
    return [
        [
            get_cts_controller(As, Bs, i, j, mode_priors)
            for i in range(n_modes)
        ]
        for j in range(n_modes)
    ]
    