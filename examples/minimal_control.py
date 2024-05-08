import logging
import numpy as np

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.algebra import extract_adjacency
from hybrid_control import observer_transition_model as otm
from hybrid_control.logisitc_reg import mode_posterior
from hybrid_control.generate_ctl_prior import generate_all_priors


logging.basicConfig(level=logging.INFO)


def control_prior(discrete_action):
    """
    Takes discrete action, maps to cts point, use lqr to generate cts action
    """
    pass


def p_0():
    return np.random.normal(np.array([0,0]), .1)


if __name__ == "__main__":
    ENV_STEPS = 10

    env = get_three_region_env(0, 0, 5)
    W = np.block([[linear.w] for linear in env.linear_systems])
    b = np.block([linear.b for linear in env.linear_systems])

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

    action = p_0()

    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        print("observation", observation)
        probs = mode_posterior(observation, W, b)
        probs = np.eye(len(probs))[np.argmax(probs)]   
        agent, discrete_action = otm.step_active_inf_agent(agent,  probs)
        control_prior(discrete_action)
        # discrete -> cts action
