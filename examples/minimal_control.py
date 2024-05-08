import numpy as np

from hybrid_control.environments.library import get_three_region_env
from hybrid_control.algebra import extract_adjacency
import hybrid_control.observer_transition_model


if __name__ == "__main__":

    env = get_three_region_env(0, 0, 5)
    Ws = np.block([[linear.w] for linear in env.linear_systems])
    bs = np.block([linear.b for linear in env.linear_systems])

    A = extract_adjacency(Ws, bs)
    # get adjacency matrix

    # Question: use adjacency or pymdp rep
    # get obs transition matrix


    #

    # get central points for modes


    # control cost for each transition - each direction