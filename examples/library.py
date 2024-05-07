import numpy as np
from .environments import SwitchSystem, LinearSystem



def get_linearly_seperated_env():
    A_1 = np.array([[0, 0], [0, 1]])
    B_1 = np.array([[1, 0], [0, 1]])
    w_1 = np.array([1, 1])

    A_2 = np.array([[-1, 0], [0, -1]])
    B_2 = np.array([[1, 0], [0, 1]])
    w_2 = np.array([-1, 1])
    return SwitchSystem(
        linear_systems=[
            LinearSystem(A_1, B_1, w_1),
            LinearSystem(A_2, B_2, w_2)
        ],
        x = np.array([0, 1])
    )


def get_three_region_env():
    A_1 = np.array([[0, 0], [0, 1]])
    B_1 = np.array([[1, 0], [0, 1]])
    w_1 = np.array([1, 1])

    A_2 = np.array([[-1, 0], [0, -1]])
    B_2 = np.array([[1, 0], [0, 1]])
    w_2 = np.array([-1, 1])

    A_3 = np.array([[-1, 1], [0, -1]])
    B_3 = np.array([[1, 0], [0, 1]])
    w_3 = np.array([0, 1])
    b_3 = 5
    return SwitchSystem(
        linear_systems=[
            LinearSystem(A_1, B_1, w_1),
            LinearSystem(A_2, B_2, w_2),
            LinearSystem(A_3, B_3, w_3, b_3)
        ],
        x = np.array([0, 1])
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import plot_phases

    env = get_three_region_env()

    Ws = np.block([[linear.w] for linear in env.linear_systems])
    bs = np.block([linear.b for linear in env.linear_systems])

    plot_phases(Ws, bs)
    plt.show()
    

