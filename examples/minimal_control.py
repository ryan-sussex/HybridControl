import logging
from typing import List
import numpy as np

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors
from hybrid_control.lqr import LinearController, convert_to_servo


logging.basicConfig(level=logging.INFO)


def control_prior(discrete_action):
    """
    Takes discrete action, maps to cts point, use lqr to generate cts action
    """
    pass


def p_0():
    return np.random.normal(np.array([0, 0]), 0.1)


def estimated_system_params(env):
    """
    Warning! Passed env for simulation, real model does not have access
    """
    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])
    As = [linear.A for linear in env.linear_systems]
    Bs = [linear.B for linear in env.linear_systems]
    return W, b, As, Bs


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
        self.agent = self\
            .get_discrete_controller(W, b)
        self.cts_ctrs = self\
            .get_all_cts_controllers(As, Bs, self.mode_priors)
        
        self.W = W
        self.b = b

    def mode_posterior(self, observation):
        return mode_posterior(observation, self.W, self.b)

    @staticmethod
    def get_discrete_controller(W, b):
        adj = extract_adjacency(W, b)
        return otm.construct_agent(adj)

    def get_cts_controller(self, As, Bs, i: int, j: int):
        """
        Constructs the controller for traversing region i to reach goal j
        """
        lc = LinearController(
            As[i],
            Bs[i],
            Q=np.eye(As[i].shape[0]) * 100,   # TODO: Magic numbers
            R=np.eye(Bs[i].shape[0]),
        )
        return convert_to_servo(lc, self.mode_priors[j])
    
    def get_all_cts_controllers(self, As, Bs, mode_priors):
        """
        Returns list of lists, where element list[i][j] is the controller
        for going from region i, to the prior specified by mode_prior[j]
        """
        return [
            [
                self.get_cts_controller(As, Bs, i, j)
                for i in range(self.n_modes)
            ]
            for j in range(self.n_modes)
        ]
    
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
        cts_prior = priors[discrete_action]
        # Continuous
        cts_ctr = self.cts_ctrs[discrete_action][idx_mode]
        x_bar = np.r_[observation - cts_prior, 1]  # internal coords TODO: simplify this
        action = cts_ctr.finite_horizon(x_bar, t=0, T=100)  # TODO: magic numbers
        return action


if __name__ == "__main__":
    ENV_STEPS = 10

    env = get_three_region_env(0, 0, 5)

    W, b, As, Bs = estimated_system_params(env)

    priors = generate_all_priors(W, b)
    print(priors)

    adj = extract_adjacency(W, b)
    print(adj)

    agent = otm.construct_agent(adj)
    print(agent.B[0][:, :, 0])

    # TODO:
    # get central points for discrete modes ()
    # calculate costs between modes
    # lift reward to pymdp agent

    controller = Controller(As=As, Bs=Bs, W=W, b=b)

    action = p_0()

    traj = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)

        action = controller.policy(observation)

        # # controller logic
        # probs = mode_posterior(observation, W, b)
        # idx_mode = np.argmax(probs)
        # mode = np.eye(len(probs))[idx_mode]

        # agent, discrete_action = otm.step_active_inf_agent(agent, mode)
        # cts_prior = priors[discrete_action]

        # active_linear = env.linear_systems[idx_mode]

        # lc = LinearController(
        #     As[idx_mode],
        #     Bs[idx_mode],
        #     Q=np.eye(active_linear.A.shape[0]) * 100,
        #     R=np.eye(active_linear.B.shape[1]),
        # )
        # lc = convert_to_servo(lc, cts_prior)

        # x_bar = np.r_[observation - cts_prior, 1]  # internal coords

        # # print(observation)
        # action = lc.finite_horizon(x_bar, t=0, T=100)

    prob_hist = [mode_posterior(x, W, b) for x in traj]
