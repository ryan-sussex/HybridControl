from typing import Optional
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are


class LinearController:
    """
    Controller for a state space model:
        x' = Ax + Bu + e

    With affine boundaries region:
        min(cx + d) = 0

    Aiming for terminal point along the boundary with quadratic cost, let
        cv + d = 0
    then cost function xP(1, 0, 0, 0)Px, where P projects onto the line
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        b: Optional[np.ndarray] = None,
        Q_f: Optional[np.ndarray] = None,
        c=None,
        d=None,
        v=None,
        x_ref=None,
    ) -> None:
        if b is None:
            b = np.zeros(A.shape[0])
        if x_ref is None:
            x_ref = np.zeros(A.shape[0])
        if Q_f is None:
            Q_f = Q
        
        print("Q_f", Q_f)
        self.b = b

        self.A, self.B, self.Q, self.R, self.Q_f = create_biased_matrices(
            A, B, Q, R, self.b, Q_f
        )
        self.x_ref = x_ref

        self.c = c
        self.d = d
        self.v = v
        # Precomputed
        self.S_ih = None
        self.Ks = None

    def coordinate_transform(fn):
        @wraps(fn)
        def wrapped(self, x, *args, **kwargs):
            if x.shape[0] == self.x_ref.shape[0]:
                x = np.r_[x - self.x_ref, 1]  # internal coords
            return fn(self, x, *args, **kwargs)

        return wrapped

    @coordinate_transform
    def infinite_horizon(self, x):
        if self.S_ih is None:
            S = solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.S_ih = S

        K = calculate_gain(self.A, self.B, self.Q, self.R, self.S_ih)
        return -K @ x

    @coordinate_transform
    def finite_horizon(self, x, t, T):
        if self.Ks is not None:
            if t >= T:
                t = T - 1
            return -self.Ks[t] @ x

        Ss = [None for _ in range(T)]
        Ss[T - 1] = np.zeros(self.A.shape)
        for i in range(T):
            Q = self.Q_f if i == 0 else self.Q
            Ss[T - i - 2] = backwards_riccati(
                self.A, self.B, Q, self.R, Ss[T - i - 1]
            )

        self.Ks = [calculate_gain(self.A, self.B, self.Q, self.R, S) for S in Ss]
        return self.finite_horizon(x, t, T)

    @coordinate_transform
    def instantaneous_cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u


def create_biased_matrices(A: np.ndarray, B: np.ndarray, Q, R, bias: np.ndarray, Q_f):
    """
    Translates the linear system to the affine system in (x-x_ref) coords
        z = [x, 1]
        z = Az + Bu

    Where
        A <- [A, b; 0, 1]
        B <- [B, 0]

    and Q is transported to the point x_ref.
    """
    A_shape = A.shape
    B_shape = B.shape

    bias = bias[:, None]
    zeros = np.zeros(A_shape[1])[:, None]
    ones = np.array([[1]])

    A = np.block([[A, bias], [zeros.T, ones]])
    B = np.block([[B], [np.zeros((1, B_shape[1]))]])

    Q_out = np.zeros(A.shape)
    Q_out[: Q.shape[0], : Q.shape[1]] = Q

    Q_f_out = np.zeros(A.shape)
    Q_f_out[: Q_f.shape[0], : Q_f.shape[1]] = Q_f
    return A, B, Q_out, R, Q_f_out


def convert_to_servo(linear_controller: LinearController, x_ref) -> LinearController:
    """
    Translates the linear system to the affine system in (x-x_ref) coords
        z = [x - x_ref, 1]
        z = Az + Bu

    Where
        A <- [A, b; 0, 1]
        B <- [B, 0]

    and Q is transported to the point x_ref.
    """
    A = linear_controller.A
    # unbiased dynamics
    A_ub = A[:-1, :-1]
    bias = (A_ub - np.eye(A_ub.shape[0])) @ x_ref
    A[:-1, -1] += bias
    linear_controller.A = A
    linear_controller.x_ref = x_ref
    return linear_controller


def calculate_gain(A, B, Q, R, S):
    return np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A


def backwards_riccati(A, B, Q, R, S):
    return (
        Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ B.T @ S @ A
    )


def get_trajectory_cost(A, B, Q, R, b, x_0, x_ref):
    T = 100  # TODO: magic number
    # TODO use constraints in simulation to calculate this cost
    lc = LinearController(A, B, Q, R)
    lc = convert_to_servo(lc, x_ref)
    accum_cost = 0
    # Simulate system
    x = x_0
    traj = [x]
    for t in range(T):
        u = lc.finite_horizon(x, t=t, T=T)
        accum_cost += lc.instantaneous_cost(x, u)
        x = A @ x + B @ u + np.random.normal(np.zeros(x.shape), scale=0.2) - b
        traj.append(x)
    return accum_cost


if __name__ == "__main__":

    T = 100

    As = [
        np.array([[0, 0], [0, 2]]),
        np.array([[-1.4, 0], [0, -2]]),
        np.array([[-1, 1], [0, -1]]),
    ]

    Bs = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 1]]),
    ]

    A = As[1]
    B = Bs[1]
    b = np.ones(A.shape[0], dtype="float") * -5

    Q = np.eye(2) * 20
    R = np.eye(2) 
    Q_f = np.eye(2) * 1000

    x_0 = np.array([0, 7.29713065])
    x_ref = np.array([2, 0])

    lc = LinearController(A, B, Q, R, b=b, Q_f=Q_f)
    lc = convert_to_servo(lc, x_ref)

    accum_cost = 0
    # Simulate system
    x = x_0
    traj = [x]
    for t in range(T):
        u = lc.finite_horizon(x, t=t, T=T)
        accum_cost += lc.instantaneous_cost(x, u)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.1) + b
        traj.append(x)

    X = np.column_stack(traj)

    av_cost = accum_cost / T
    print("total control cost", accum_cost)
    print("average control cost per timestep", av_cost)

    # Plots
    ax1 = plt.subplot(111)
    for i in range(X.shape[1]):
        ax1.plot(
            X[0, i],
            X[1, i],
            color=(0.3, 0.7, i / X.shape[1]),
            marker="o",
            linestyle="none",
        )
    print("x final", X[:, -2])
    plt.show()
