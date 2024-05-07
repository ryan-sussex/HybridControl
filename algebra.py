from typing import List
import logging
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

logger = logging.getLogger()

def extract_adjacency(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Creates and adjacency matrix from the logistic regression parameters
        1. For each w_i find the halfspace LMI representation
        2. For this representation check which constraints are redundant
        3. Adjacency matrix is formed from non-redundant transitions.
    """
    # For each w_i we have halfspace polytope rep
    # Find redundant inequalities by checking lp
    adjacency = []
    for i in range(len(W)):
        logger.info(f"Checking node {str(i)}..")
        W_, b_ = get_polytope_rep(W, b, i)
        redundants = check_for_redundancy(W_, b_)
        adjacency.append([int(not redundant) for redundant in redundants])
    return np.array(adjacency)


def get_polytope_rep(W: np.ndarray, b: np.ndarray, i):
    """
    Considering softmax(Wx + b)
    
    For a given i (discrete region), finds the system of inequalities
        W_ x <= b_
    governing the region for which i is active, (largest in the softmax)
    """
    w_mode = W[i]
    b_mode = b[i]
    W_ = W - w_mode
    b_ = b_mode - b
    return W_, b_


def check_for_redundancy(W, b):
    redundant = []
    for i in range(len(W)):
        res = linprog(W[i], A_ub=W, b_ub=b)
        if res["status"] == 0:
            is_redundant = (np.abs(res["fun"] - b[i]).sum() > 0.1)
            logger.info(f"..linear program constraint check found {str(i)}, redundant {is_redundant}")
            redundant.append(is_redundant)
        elif res["status"] == 3:
            logger.info(f"..linear program constraint check is unbounded for {str(i)}")
            redundant.append(False)
        else:
            logger.info(f"..Linear program constraint check failed for node {str(i)}")
            redundant.append(True)
    return redundant


def get_hplane_intersection(
    w_1: np.ndarray, w_2: np.ndarray, b_1: float = 0, b_2: float = 0
):
    """
    N-dim Hyperplanes defined by w dot x = 0

    Find n-1 plane of intersection.
    """
    W = np.block([[w_1], [w_1 - w_2]])
    return null_space(W)


def project(u, v):
    return u - (u @ v) / (v @ v) * v


def get_basis_vecs(c):
    """
    Find basis for the plane
    """
    n_dims = c.shape[0]
    std_basis = np.eye(n_dims)
    c = c / (c @ c)
    basis = [c]
    for i in range(n_dims - 1):
        vec = project(std_basis[i], basis[i])
        vec = vec / (vec @ vec)
        basis.append(vec)
    return basis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import matplotlib.pyplot as plt
    from examples.utils import plot_phases
    from examples.library import get_three_region_env

    env = get_three_region_env(0, 0, 5)
    
    Ws = np.block([[linear.w] for linear in env.linear_systems])
    bs = np.block([linear.b for linear in env.linear_systems])



    # print(bs)
    # # W = np.random.randint(-10, 10, (10, 3))
    # # b = np.random.randint(-10, 10, (10, 1))

    A = extract_adjacency(Ws, bs)
    print(A)

    # plot_phases(Ws, bs)
    # plt.show()

    # import matplotlib.pyplot as plt
    # import numpy as np

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    print(X[0][0])
    # Z = np.sin(R)

    # Plot the surface.
    Z1 = Ws[0][0] * X  + Ws[0][1] * Y + bs[0]
    # surf = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    Z2 = Ws[1][0] * X  + Ws[1][1] * Y +  bs[1]
    # surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm,
    #                         linewidth=0, antialiased=False)
    
    Z3 = Ws[2][0] * X  + Ws[2][1] * Y +  bs[2]
    # surf = ax.plot_surface(X, Y, Z3, cmap=cm.coolwarm,
    #                             linewidth=0, antialiased=False)
    
    all = np.stack([Z1, Z2, Z3])
    # print(all.shape)
    Z = np.max(all, axis=0)
    # print(Z.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()