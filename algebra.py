from typing import List
import numpy as np


def extract_adjacency(W: np.ndarray, b:np.ndarray) -> np.ndarray:
    pass


def get_hplane_intersection(w_1, w_2):
    """
    N-dim Hyperplanes defined by w dot x = 0

    Find n-1 plane of intersection.
    """
    return 


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
    pass