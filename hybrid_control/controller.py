import logging
from typing import List, Optional
import numpy as np
from pymdp import utils as pu


from ssm import SLDS

from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo
from hybrid_control.costs import get_cost_matrix, get_prior_over_policies


logger = logging.getLogger("controller")


Q_SCALE = 100
R_SCALE = 1
LQR_HORIZON = 5


class Controller:

    def __init__(
        self,
        As: List[np.ndarray],
        Bs: List[np.ndarray],
        bs: List[np.ndarray],
        W_u: np.ndarray,
        W_x: np.ndarray,
        b: np.ndarray,
        rslds: Optional[SLDS] = None,
        reward_pos_cts: Optional[np.ndarray] = None,
        max_reward: Optional[float] = None,
        max_u: float = np.inf,
        min_u: float = -np.inf,
        Sigmas: Optional[List[np.ndarray]]=None
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
        self.cts_ctrs = get_all_cts_controllers(As, Bs, bs, self.mode_priors)

        self.As = As
        self.Bs = Bs
        self.bs = bs

        self.W_x = W_x
        self.W_u = W_u
        self.b = b
        
        self.adj = extract_adjacency(W_x=W_x, W_u=W_u, b=b)
        self.agent = get_discrete_controller(self.adj, rwd_idx=None)
        # Will be overwritten if reward is passed

        self._reward_pos_cts = None
        self.reward_pos_dsc = None
        self.final_controller = None
        self.max_reward = max_reward
        self.reward_pos_cts = reward_pos_cts

        self.cost_matrix = get_cost_matrix(
            self.adj,
            self.mode_priors,
            As,
            Bs,
            **get_default_lqr_costs(self.obs_dim, self.action_dim),
            bs=bs,
        )
        self.discrete_action = 0
        self._rslds: Optional[SLDS] = rslds  # Store rslds for convenicence
        self.max_u = max_u
        self.min_u = min_u
        self.prev_mode: Optional[int] = None
        self.Sigmas = Sigmas
        if Sigmas is not None:
            for i, sigma in enumerate(Sigmas):
                print(sigma)
                std = sigma.dot(sigma)
                logger.info(f"std for mode {i}:{std}")

    @property
    def reward_pos_cts(self):
        return self._reward_pos_cts

    @reward_pos_cts.setter
    def reward_pos_cts(self, obs):
        self._reward_pos_cts = obs
        # Set other attributes to keep in sync
        self.reward_pos_dsc = self._get_reward_idx(obs)
        self.agent = get_discrete_controller(self.adj, self.reward_pos_dsc)
        if self.reward_pos_cts is not None:
            self.final_controller = get_final_controller(
                self.As, self.Bs, self.bs, self.reward_pos_cts, self.reward_pos_dsc
            )
        logger.info(
            f"setting reward at continuous:{self.reward_pos_cts} "
            f"and discrete:{self.reward_pos_dsc}"
        )

    def set_known_reward(self, reward, pos):
        self.max_reward = reward
        self.reward_pos_cts = pos

    def mode_posterior(self, observation, action: Optional[np.ndarray] = None):
        if action is None:
            logger.warning(
                "Recieved None, for action.. ..setting action to zeros, "
                "note this may lead to inaccurate inference."
            )
            action = np.zeros(self.action_dim)
        return mode_posterior(observation, action, self.W_x, self.W_u, self.b)

    def _get_reward_idx(
        self, reward_pos: Optional[np.ndarray], action: Optional[np.ndarray] = None
    ):
        if reward_pos is None:
            return None
        return np.argmax(self.mode_posterior(reward_pos), action)

    def _check_and_update_reward(
        self,
        observation: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None,
    ):
        logger.debug("  checking reward..")
        if (self.max_reward is None) or (reward > self.max_reward):
            self.max_reward = reward
            logger.info( f"Found larger reward:{self.max_reward}")
            self.reward_pos_cts = observation
        return

    def policy(
        self,
        observation: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[np.ndarray] = None
    ):
        """
        Takes a continuous observation, outputs continuous action.
        """
        logger.debug("Executing policy..")
        if observation is None:
            logger.info("No observation, returning default action.")
            return self.p_0(self.action_dim)

        if reward is not None:
            self._check_and_update_reward(observation, action, reward)

        probs = self.mode_posterior(observation, action)
        idx_mode = np.argmax(probs)
        logger.debug(f"  Inferred mode {idx_mode}")

        if self.prev_mode is None:
            is_connected = np.sum(self.adj, axis=1) > 1
            is_connected = self.adj[idx_mode] == 1
            is_connected[idx_mode] = False
            print(is_connected)
            sigmas = [sigma.dot(sigma) + np.random.normal(0, 1) if is_connected[i] else - np.inf for i, sigma in enumerate(self.Sigmas) ]
            # break ties arbritrarily
            self.discrete_action = np.argmax(sigmas)
            logger.info(f"first obs, picking action {self.discrete_action} based on uncertainty")


        if (self.prev_mode is not None) and (idx_mode != self.prev_mode):
            print(self.prev_mode)
            logger.info(f"  Inferred mode {idx_mode}")
            logger.info("Entered new mode, triggering discrete planner")
            if idx_mode == self.discrete_action:
                logger.info(
                    f"  Discrete Goal {self.agent.mode_action_names[self.discrete_action]} Achieved!"
                )

            obs = pu.to_obj_array(probs)

            # Discrete
            self.agent.E = get_prior_over_policies(
                self.adj, self.agent, self.cost_matrix, idx_mode
            )
            self.agent, discrete_action = otm.step_active_inf_agent(
                self.adj, idx_mode, self.agent, obs
            )
            cts_prior = self.mode_priors[discrete_action]
            self.discrete_action = discrete_action  # For debugging
            logger.info(f"  Aiming for {cts_prior}")
            logger.info(f"  max reward @ {self.reward_pos_dsc} @ {self.reward_pos_cts}")

        if (idx_mode == self.reward_pos_dsc) and (self.discrete_action == idx_mode):
            logger.info("Attempting to stabilise at max reward")
            action = self.final_controller.finite_horizon(observation, t=0, T=LQR_HORIZON)
            action = bound_action(action, self.max_u, self.min_u)
            logger.info(f" ..Returning action {action}")
            return action
        
        self.prev_mode = idx_mode
        # Continuous
        cts_ctr = self.cts_ctrs[self.discrete_action][idx_mode]
        # x_bar = np.r_[observation - cts_prior, 1]  # internal coords TODO: simplify this
        action = cts_ctr.finite_horizon(observation, t=0, T=LQR_HORIZON)  # TODO: magic numbers
        action = bound_action(action, self.max_u, self.min_u)
        logger.debug(f" ..Returning action {action}")
        return action

    @staticmethod
    def p_0(action_dim: int):
        """
        Default action before observations recieved.
        """
        return np.random.normal(np.zeros(action_dim), 6)

    def estimate(self, obs, actions, **kwargs):
        logger.info("re-estimation..")
        rslds = _estimate(
            obs, actions, self.obs_dim, self.action_dim, self.n_modes, **kwargs
        )
        self._rslds = rslds
        return estimated_system_params(rslds)

    def estimate_and_identify(self, obs, actions, **kwargs):
        param_dct = self.estimate(obs, actions, rslds=self._rslds, **kwargs)
        return Controller(
            **param_dct,
            rslds=self._rslds,
            reward_pos_cts=self.reward_pos_cts,
            max_reward=self.max_reward,
            max_u=self.max_u,
            min_u=self.min_u
        )


def bound_action(action, max_u, min_u):
    """only works for 1d actions"""
    action[action > max_u] = max_u
    action[action < min_u] = min_u
    return action


def get_discrete_controller(adj, rwd_idx: Optional[int]):
    return otm.construct_agent(adj, rwd_idx=rwd_idx)


def get_default_lqr_costs(obs_dims, action_dims):
    """
    Return
    -------
    Dict: (Q, R)
    """
    return dict(Q=np.eye(obs_dims) * Q_SCALE, R=np.eye(action_dims) * R_SCALE)  # TODO: Magic numbers


def get_cts_controller(As, Bs, bs, i: int, j: int, mode_priors: List):
    """
    Constructs the controller for traversing region i to reach goal j
    """
    lc = LinearController(
        As[i], Bs[i], b=bs[i], **get_default_lqr_costs(As[i].shape[0], Bs[i].shape[1])
    )
    return convert_to_servo(lc, mode_priors[j])


def get_all_cts_controllers(As, Bs, bs, mode_priors: List):
    """
    Returns list of lists, where element list[i][j] is the controller
    for going from region i, to the prior specified by mode_prior[j]
    """
    n_modes = len(mode_priors)
    return [
        [get_cts_controller(As, Bs, bs, i, j, mode_priors) for i in range(n_modes)]
        for j in range(n_modes)
    ]


def get_final_controller(As, Bs, bs, reward_pos_cts, reward_pos_discrete):
    lc = LinearController(
        As[reward_pos_discrete],
        Bs[reward_pos_discrete],
        # b=bs[reward_pos_discrete],
        **get_default_lqr_costs(
            As[reward_pos_discrete].shape[0], Bs[reward_pos_discrete].shape[1]
        ),
    )
    return convert_to_servo(lc, reward_pos_cts)


def estimated_system_params(rslds: SLDS):
    dynamic_params = rslds.dynamics.params
    emission_params = rslds.emissions.params
    softmax_params = rslds.transitions.params

    W_u, W_x, b = softmax_params
    As, bs, Bs, Sigmas = dynamic_params
    # TODO: bias term for linear ctrlrs, and extra weight for inputs
    # Workout exactly what Sigmas are
    return dict(W_u=W_u, W_x=W_x, b=b, As=As, Bs=Bs, bs=bs, Sigmas=Sigmas)


def _estimate(
    obs,
    actions,
    d_obs,
    d_actions,
    k_components,
    n_iters: int = 100,
    rslds: Optional[SLDS] = None,
) -> SLDS:
    """
    Fits an RSLDS, if rslds is passed will just call fit function,
    otherwise randomly reinitalise
    """
    if rslds is None:
        logger.info("No existing RSLDS found, initialising..")
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


def get_initial_controller(d_obs, d_actions, k_components, **kwargs):
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
    return Controller(**estimated_system_params(rslds), rslds=rslds, **kwargs)
