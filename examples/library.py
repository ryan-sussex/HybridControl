import numpy as np
from environments import SwitchSystem, LinearSystem



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
            LinearSystem(A_2, B_2)
        ],
        x = np.array([0, 1])
    )



if __name__ == "__main__":
    env = get_linearly_seperated_env()

    for _ in range(10):
        res = env.step(u=np.array([0, 1]))
        print(res)


    