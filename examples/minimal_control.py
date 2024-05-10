import logging
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

    action = p_0()

    traj = []
    for i in range(ENV_STEPS):
        observation, reward, terminated, truncated, info = env.step(action)
        traj.append(observation)
        
        # controller logic
        probs = mode_posterior(observation, W, b)
        idx_mode = np.argmax(probs)
        mode = np.eye(len(probs))[idx_mode]   

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


    prob_hist = [mode_posterior(x, W, b) for x in traj]