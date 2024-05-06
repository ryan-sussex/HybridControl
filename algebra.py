from typing import List
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog


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
    b_ = b - b_mode
    return W_, b_


def check_for_redundancy(W, b):
    redundant = []
    for i in range(len(W)):
        res = linprog(W[i], A_ub=-W, b_ub=-b)
        if res["status"] == 0:
            redundant.append((np.abs(res["fun"] - b[i]).sum() > 0.001))
        else:
            redundant.append(False)
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

    W = np.random.randint(-10, 10, (10, 3))
    b = np.random.randint(-10, 10, (10, 1))

    A = extract_adjacency(W, b)

    print(A)
