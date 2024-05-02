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

    def __init__(self, A, B, Q, R, c=None, d=None, v=None) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.c = c
        self.d = d
        self.v = v
        # Precomputed
        self.S_ih = None
        self.Ks = None

    @staticmethod
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
            # assert np.abs(vec @ c) < .01
        return basis

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


def convert_to_servo(linear_controller: LinearController) -> LinearController:
    A_shape = linear_controller.A.shape
    B_shape = linear_controller.B.shape
    A = np.block(
        [
            [linear_controller.A, np.zeros(A_shape)],
            [np.eye(A_shape[0]), np.zeros(A_shape)]
        ]
    )
    B = np.block(
        [[linear_controller.B], [np.zeros(B_shape)]]
    )
    Q = np.block(
        [
            [np.zeros(A_shape), np.zeros(A_shape)],
            [np.zeros(A_shape), linear_controller.Q]
        ]
    )
    # print(Q)
    # # R = np.block(
    # #         [[linear_controller.R], [np.zeros(B_shape)]]
    # # )
    # print(A.shape)
    # print(B.shape)
    # print(Q.shape)
    return LinearController(A, B, Q, linear_controller.R)


def calculate_gain(A, B, Q, R, S):
    return np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A


def backwards_riccati(A, B, Q, R, S):
    return (
        Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ B.T @ S @ A
    )


def project(u, v):
    return u - (u @ v) / (v @ v) * v


if __name__ == "__main__":

    A = np.array([[-1, 1], [2, 1]])
    B = np.array([[1, 2], [0.1, 3]])

    Q = np.eye(2) * 100

    lc = LinearController(A, B, Q, Q)

    lc = convert_to_servo(lc)

    x_0 = np.array([0, 6])
    
    x_ref = None
    # x_ref = np.array([-3, 4])
    # plt.figure(figsize=(6, 6))

    x = x_0
    traj = [x]
    e = x - x_0
    x_bar = np.r_[x, e]
    for _ in range(100):
        u = lc.infinite_horizon(x_bar)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
        traj.append(x)
    print(x)
    X = np.column_stack(traj)


    ax1 = plt.subplot(111)
    for i in range(X.shape[1]):
        ax1.plot(
            X[0, i],
            X[1, i],
            color=(0.5, 0.2, i / X.shape[1]),
            marker="x",
            linestyle="none",
        )

    # x = x_0
    # traj = [x]
    # for t in range(100):
    #     u = lc.finite_horizon(x, t=t, T=6, x_ref=x_ref)
    #     x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
    #     traj.append(x)
    # X = np.column_stack(traj)

    # # ax2 = plt.subplot(111)
    # for i in range(X.shape[1]):
    #     ax1.plot(
    #         X[0, i],
    #         X[1, i],
    #         color=(0.3, 0.7, i / X.shape[1]),
    #         marker="o",
    #         linestyle="none",
    #     )

    plt.show()
