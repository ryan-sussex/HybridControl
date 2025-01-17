from typing import List
import logging
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from hybrid_control.config import param_config

logger = logging.getLogger("linear_algebra")


UPPER_BOUND = param_config.getint("UPPER_BOUND", 5)
LOWER_BOUND = param_config.getint("LOWER_BOUND", 5)


def extract_adjacency(W_x: np.ndarray, W_u: np.ndarray, b: np.ndarray, u_max=1):    # TODO: get from env
    """
    Constructs adjacency matrix from decision boundaries.

    Parameters
    ----------
        W_x: (n_modes, state_dim)
        W_u: (n_modes, control_dim)
        b: (n_modes)
    """
    W = np.c_[W_x, W_u]
    if u_max is not None:
        u_idx = np.zeros((1, W.shape[1]))
        u_idx[- W_u.shape[0]:] = 1
        W = np.r_[W, u_idx, -u_idx]
        b = np.r_[b, u_max, -u_max]
    adj = _extract_adjacency(W, b)
    
    if u_max is not None:
        adj = adj[:-2, :-2]
    return adj


def _extract_adjacency(W: np.ndarray, b: np.ndarray) -> np.ndarray:
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
        logger.info(f"Checking node i={str(i)}..")
        W_, b_ = get_polytope_rep(W, b, i)
        redundants = check_for_redundancy(W_, b_)
        redundants.insert(i, False)
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
    W = np.delete(W, (i), axis=0)
    b = np.delete(b, (i), axis=0)
    W_ = W - w_mode
    b_ = b_mode - b
    return W_, b_


def check_for_redundancy(W, b):
    redundant = []
    for i in range(len(W)):
        # -ve because lingprog minimises, we need max
        res = linprog(-W[i], A_ub=W, b_ub=b, bounds=(LOWER_BOUND, UPPER_BOUND))
        if res["status"] == 0:
            # print(np.abs(res["fun"] - b[i]).sum())
            is_redundant = np.abs(res["fun"] + b[i]).sum() > 0.1
            logger.info(
                f"..linear program constraint check found j={str(i)},"
                f"redundant {is_redundant}"
            )
            redundant.append(is_redundant)
        elif res["status"] == 3:
            logger.info(
                "..linear program constraint check " f"is unbounded for j={str(i)}"
            )
            redundant.append(False)
        else:
            status = res["status"]
            logger.info(
                "..Linear program constraint check failed"
                f"for node j={str(i)} with code {status}"
            )
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
    state_dim = 4
    control_dim = 3
    n_modes = 5
    W_x = np.ones((n_modes, state_dim))
    W_u = np.ones((n_modes, control_dim))
    b = np.zeros(5)
    extract_adjacency(W_u=W_u, W_x=W_x, b=b)
