import logging
from typing import List, Optional
import numpy as np

from ssm import SLDS

from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo
from hybrid_control.costs import get_cost_matrix, get_prior_over_policies


logger = logging.getLogger("controller")


class Controller:

    def __init__(
        self,
        As: List[np.ndarray],
        Bs: List[np.ndarray],
        bs: List[np.ndarray],
        W_u: np.ndarray,
        W_x: np.ndarray,
        b: np.ndarray,
    ):
        """
        Parameters
        -----------
        As, Bs, bs: List of arrays (each of same shape)
            The parameters associated to each linear system i.e.
            x' = A[i] @ x + B[i] @ u + b[i]

        W_u, W_x, b: Arrays with first dim as the number components
            The parameters for inducing switches i.e.
            softmax(W_u @ u + W_x @ x + b)
        """
        self.obs_dim = As[0].shape[0]
        self.action_dim = Bs[0].shape[1]
        self.n_modes = len(As)

        if W_u is None:
            W_u = np.zeros((self.n_modes, self.action_dim))
        if bs is None:
            bs = [np.zeros((self.obs_dim)) for _ in range(self.n_modes)]

        self.mode_priors = generate_all_priors(W_x, b)
        self.agent = get_discrete_controller(W_x, b)
        self.cts_ctrs = get_all_cts_controllers(As, Bs, self.mode_priors)
        self.W_x = W_x
        self.b = b

        self.adj = extract_adjacency(W_x, b)
        self.cost_matrix = get_cost_matrix(
            self.adj,
            self.mode_priors,
            As,
            Bs,
            **get_default_lqr_costs(self.obs_dim, self.action_dim)
        )

    def mode_posterior(self, observation):
        return mode_posterior(observation, self.W_x, self.b)

    def policy(self, observation: Optional[np.ndarray] = None):
        """
        Takes a continuous observation, outputs continuous action.
        """
        if observation is None:
            logger.info("No observation, returning default action.")
            return self.p_0(self.action_dim)

        probs = self.mode_posterior(observation)
        idx_mode = np.argmax(probs)
        mode = np.eye(len(probs))[idx_mode]  # one-hot rep
        logger.debug(f"Inferred mode {mode}")

        # Discrete
        self.agent.E = get_prior_over_policies(
            self.agent, self.cost_matrix, idx_mode, alpha=0.0001    # TODO: magic number
        )
        self.agent, discrete_action = otm.step_active_inf_agent(self.agent, mode)
        cts_prior = self.mode_priors[discrete_action]

        # Continuous
        cts_ctr = self.cts_ctrs[discrete_action][idx_mode]
        x_bar = np.r_[observation - cts_prior, 1]  # internal coords TODO: simplify this
        action = cts_ctr.finite_horizon(x_bar, t=0, T=100)  # TODO: magic numbers
        return action

    @staticmethod
    def p_0(action_dim: int):
        """
        Default action before observations recieved.
        """
        return np.random.normal(np.zeros(action_dim), 0.1)

    def estimate(self, obs, actions, **kwargs):
        logger.info("re-estimation..")
        rslds = _estimate(
            obs, actions, self.obs_dim, self.action_dim, self.n_modes, **kwargs)
        return estimated_system_params(rslds)

    def estimate_and_identify(cls, obs, actions, **kwargs):
        param_dct = cls.estimate(obs, actions, **kwargs)
        return Controller(**param_dct)


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


def estimated_system_params(rslds: SLDS):
    """
    Warning! Passed env for simulation, real model does not have access
    """
    dynamic_params = rslds.dynamics.params
    emission_params = rslds.emissions.params
    softmax_params = rslds.transitions.params

    W_u, W_x, b = softmax_params
    As, bs, Bs, Sigmas = dynamic_params
    # TODO: bias term for linear ctrlrs, and extra weight for inputs
    # Workout exactly what Sigmas are
    return dict(
        W_u=W_u, W_x=W_x, b=b, As=As, Bs=Bs, bs=bs
    )


def _estimate(obs, actions, d_obs, d_actions, k_components, n_iters: int = 100) -> SLDS:
    rslds = SLDS(
        d_obs,
        k_components,
        d_obs,
        M=d_actions,  # Control dim
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian_id",
        single_subspace=True,
    )

    rslds.initialize(obs, inputs=actions)

    q_elbos, q = rslds.fit(
        obs,
        inputs=actions,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        initialize=False,
        num_iters=n_iters,
        alpha=0.0,
    )
    # results = dict(
    #     rslds=rslds,
    #     obs=obs,
    #     q_elbos=q_elbos,
    #     q=q
    # )
    return rslds
