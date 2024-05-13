import logging
from typing import List
import numpy as np
from pymdp import utils as pu


from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo
from hybrid_control.costs import get_cost_matrix, get_prior_over_policies


logging.basicConfig(level=logging.INFO)


class Controller:

    def __init__(
        self, As: List[np.ndarray], Bs: List[np.ndarray], W: np.ndarray, b: np.ndarray
    ):
        self.n_modes = len(As)
        self.mode_priors = generate_all_priors(W, b)
        self.agent = get_discrete_controller(W, b)
        self.cts_ctrs = get_all_cts_controllers(As, Bs, self.mode_priors)
        self.W = W
        self.b = b
        self.obs_dim = As[0].shape[0]
        self.action_dim = Bs[0].shape[1]
        self.adj = extract_adjacency(W, b)
        self.cost_matrix = get_cost_matrix(
            self.adj,
            self.mode_priors,
            As,
            Bs,
            **get_default_lqr_costs(self.obs_dim, self.action_dim)
        )

    def mode_posterior(self, observation):
        return mode_posterior(observation, self.W, self.b)

    def policy(self, observation, init_step):
        """
        Takes a continuous observation, outputs continuous action.
        """
        probs = self.mode_posterior(observation)
        idx_mode = np.argmax(probs)
        mode = np.eye(len(probs))[idx_mode]  # one-hot rep
        
        obs = pu.to_obj_array(probs)
        # Discrete
        self.agent.E = get_prior_over_policies(
            self.agent, self.cost_matrix, idx_mode, alpha=0.0001    # TODO: magic number
        )
        self.agent, discrete_action = otm.step_active_inf_agent(self.agent, obs, init_step)
        cts_prior = self.mode_priors[discrete_action]
        # Continuous
        cts_ctr = self.cts_ctrs[discrete_action][idx_mode]
        x_bar = np.r_[observation - cts_prior, 1]  # internal coords TODO: simplify this
        action = cts_ctr.finite_horizon(x_bar, t=0, T=100)  # TODO: magic numbers
        return action


def get_discrete_controller(W, b):
    adj = extract_adjacency(W, b)
    return otm.construct_agent(adj)


def get_default_lqr_costs(obs_dims, action_dims):
    """
    Return
    -------
    Dict: (Q, R)
    """
    return dict(Q=np.eye(obs_dims) * 100, R=np.eye(action_dims))  # TODO: Magic numbers


def get_cts_controller(As, Bs, i: int, j: int, mode_priors: List):
    """
    Constructs the controller for traversing region i to reach goal j
    """
    lc = LinearController(
        As[i], Bs[i], **get_default_lqr_costs(As[i].shape[0], Bs[i].shape[0])
    )
    return convert_to_servo(lc, mode_priors[j])


def get_all_cts_controllers(As, Bs, mode_priors: List):
    """
    Returns list of lists, where element list[i][j] is the controller
    for going from region i, to the prior specified by mode_prior[j]
    """
    n_modes = len(mode_priors)
    return [
        [get_cts_controller(As, Bs, i, j, mode_priors) for i in range(n_modes)]
        for j in range(n_modes)
    ]
