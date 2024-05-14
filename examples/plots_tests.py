import matplotlib.pyplot as plt

from hybrid_control.plotting.utils import plot_most_likely_dynamics
from hybrid_control.controller import get_initial_controller


if __name__ == "__main__":

    K = 5  # would be unknown
    OBS_DIM = 2
    ACT_DIM = 1
    N_ITER = 100
    N_STEPS = 100

    controller = get_initial_controller(OBS_DIM, ACT_DIM, K)


    plot_most_likely_dynamics(controller)

    plt.show()