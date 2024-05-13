from typing import Optional
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
        c=None,
        d=None,
        v=None,
    ) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.b = b
        if b is not None:
            self.A, self.B, self.Q, self.R = create_biased_matrices(A, B, b)
        self.c = c
        self.d = d
        self.v = v
        # Precomputed
        self.S_ih = None
        self.Ks = None

    def infinite_horizon(self, x):
        if self.S_ih is None:
            S = solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.S_ih = S

        K = calculate_gain(self.A, self.B, self.Q, self.R, self.S_ih)
        return -K @ x

    def finite_horizon(self, x, t, T):
        if self.Ks is not None:
            if t >= T:
                t = T - 1
            return -self.Ks[t] @ x

        Ss = [None for _ in range(T)]
        Ss[T - 1] = np.zeros(self.A.shape)
        for i in range(T):
            Ss[T - i - 2] = backwards_riccati(
                self.A, self.B, self.Q, self.R, Ss[T - i - 1]
            )

        self.Ks = [calculate_gain(self.A, self.B, self.Q, self.R, S) for S in Ss]
        return self.finite_horizon(x, t, T)


def create_biased_matrices(A: np.ndarray, B: np.ndarray, bias: np.ndarray, Q, R):
    A_shape = A.shape
    B_shape = B.shape

    zeros = np.zeros(A_shape[1])[:, None]
    ones = np.array([[1]])

    A = np.block([[A, bias], [zeros.T, ones]])
    B = np.block([[B], [np.zeros((1, B_shape[1]))]])

    Q = np.zeros(A.shape)
    Q[: Q.shape[0], : Q.shape[1]] = Q
    return A, B, Q, R


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
    A_shape = linear_controller.A.shape

    bias = (linear_controller.A - np.eye(A_shape[0])) @ x_ref
    bias = bias[:, None]

    if linear_controller.b is not None:
        bias += linear_controller.b

    A, B, Q, R = create_biased_matrices(
        linear_controller.A,
        linear_controller.B,
        bias,
        linear_controller.Q,
        linear_controller.R,
    )
    return LinearController(A, B, Q, R)


def calculate_gain(A, B, Q, R, S):
    return np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A


def backwards_riccati(A, B, Q, R, S):
    return (
        Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ B.T @ S @ A
    )


def instantaneous_cost(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def get_trajectory_cost(A, B, Q, R, x_0, x_ref):

    T = 100

    lc = LinearController(A, B, Q, R)
    lc = convert_to_servo(lc, x_ref)

    accum_cost = 0

    # Simulate system
    x = x_0
    x_bar = np.r_[x - x_ref, 1]  # internal coords
    traj = [x]
    for t in range(T):
        u = lc.finite_horizon(x_bar, t=t, T=T)
        accum_cost += instantaneous_cost(x_bar, u, lc.Q, lc.R)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
        x_bar = np.r_[x - x_ref, 1]  # translate to internal coords
        traj.append(x)

    X = np.column_stack(traj)

    # av_cost = accum_cost / T

    return accum_cost


if __name__ == "__main__":

    T = 100

    As = [
        np.array([[1, 0], [0, 1]]),
        np.array([[-1, 1], [0, -1]]),
        np.array([[-1, 1], [0, -1]]),
    ]

    Bs = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 1]]),
    ]

    # A = np.array([[-.1, .1], [.2, .1]])
    # B = np.array([[1, 2], [0.1, 3]])

    A = As[1]
    B = Bs[1]

    Q = np.eye(2) * 100
    R = np.eye(2)

    x_0 = np.array([10, 7.29713065])
    x_ref = np.array([100, 0])

    lc = LinearController(A, B, Q, R)
    lc = convert_to_servo(lc, x_ref)

    accum_cost = 0

    # Simulate system
    x = x_0
    x_bar = np.r_[x - x_ref, 1]  # internal coords
    traj = [x]
    for t in range(T):
        u = lc.finite_horizon(x_bar, t=t, T=T)
        accum_cost += instantaneous_cost(x_bar, u, lc.Q, lc.R)
        x = A @ x + B @ u 
        # + np.random.normal([0, 0], scale=0.2)
        x_bar = np.r_[x - x_ref, 1]  # translate to internal coords
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
    print(x)
    plt.show()
