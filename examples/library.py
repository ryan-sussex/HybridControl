import numpy as np

from examples.environments import SwitchSystem, LinearSystem



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


def get_three_region_env(b_1=None, b_2=None, b_3=None):
    A_1 = np.array([[0, 0], [0, 1]])
    B_1 = np.array([[1, 0], [0, 1]])
    w_1 = np.array([0, -1])
    if b_1 is None:
        b_1 = 0

    A_2 = np.array([[-1, 0], [0, -1]])
    B_2 = np.array([[1, 0], [0, 1]])
    w_2 = np.array([0, 0])
    if b_2 is None:
        b_2 =0 

    A_3 = np.array([[-1, 1], [0, -1]])
    B_3 = np.array([[1, 0], [0, 1]])
    w_3 = np.array([0, 1])
    if b_3 is None:
        b_3 = -3
    return SwitchSystem(
        linear_systems=[
            LinearSystem(A_1, B_1, w_1, b_1),
            LinearSystem(A_2, B_2, w_2, b_2),
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
    

