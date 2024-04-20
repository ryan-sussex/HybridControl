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

    def infinite_horizon(self, x, x_ref=None):
        if self.S_ih is None:
            S = solve_discrete_are(self.A, self.B, self.Q, self.R)
            self.S_ih = S

        K = calculate_gain(self.A, self.B, self.Q, self.R, self.S_ih)

        if x_ref is None:
            x_ref = np.array([0, 0])
        return -K @ (x - x_ref)

    def finite_horizon(self, x, t, T, x_ref=None):
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


def calculate_gain(A, B, Q, R, S):
    return np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A


def backwards_riccati(A, B, Q, R, S):
    return (
        Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ B.T @ S @ A
    )


def project(u, v):
    return u - (u @ v) / (v @ v) * v


if __name__ == "__main__":

    A = np.array([[0, 1], [0, 1]])
    B = np.array([[1, 0], [1, 1]])

    Q = np.eye(2) * 100

    lc = LinearController(A, B, Q, Q)

    x_0 = np.array([0, 6])
    print(lc.infinite_horizon(x_0))

    x_ref = None
    # x_ref = np.array([-3, 4])
    # plt.figure(figsize=(6, 6))
    x = x_0
    traj = [x]
    for _ in range(100):
        u = lc.infinite_horizon(x, x_ref=x_ref)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
        traj.append(x)
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

    x = x_0
    traj = [x]
    for t in range(100):
        u = lc.finite_horizon(x, t=t, T=1, x_ref=x_ref)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
        traj.append(x)
    X = np.column_stack(traj)

    # ax2 = plt.subplot(111)
    for i in range(X.shape[1]):
        ax1.plot(
            X[0, i],
            X[1, i],
            color=(0.3, 0.7, i / X.shape[1]),
            marker="o",
            linestyle="none",
        )

    plt.show()

    # def plot_boundary(f, ax=None, line_only: bool = True):
    #     scale = 5
    #     x_1 = np.linspace(-scale, scale, 20)
    #     x_2 = np.linspace(-scale, scale, 20)
    #     X, Y = np.meshgrid(x_1, x_2)
    #     xy = np.column_stack((X.ravel(), Y.ravel()))
    #     f_xy = np.array(list(map(f, xy)))
    #     f_xy_norm = (f_xy - f_xy.min()) / (f_xy.max() - f_xy.min())
    #     for i in range(len(xy)):
    #         if line_only and (f_xy_norm[i]- f_xy_norm.min()) > .01:
    #             continue
    #         ax.plot(
    #             xy[i, 0],
    #             xy[i, 1],
    #             marker="o",
    #             color=(0.1, 0.2, f_xy_norm[i]),
    #             alpha=1 if line_only else f_xy_norm[i],
    #             linestyle="none",
    #         )
    #     return

    # plt.figure(figsize=(6, 6))
    # ax1 = plt.subplot(111)

    # d = np.array([1, 1])
    # c = np.array([-1, 1])
    # f = lambda x: (x - d) @ c
    # plot_boundary(f, ax=ax1, line_only=True)

    # basis = LinearController.get_basis_vecs(c)
    # k = 34
    # pt = k * basis[1] - d

    # P = np.array(basis)
    # def cost(P, x):
    #     P_ = P @ x.T
    #     Q = np.eye(x.shape[0]) * 10
    #     Q[0, 0] = 0
    #     return P_.T @ Q @ P_

    # print(
    #     cost(P, np.array([-1, 1]))
    # )

    # plt.figure(figsize=(6, 6))
    # ax2 = plt.subplot(111)
    # cst = lambda x: cost(P, x)
    # plot_boundary(cst, ax=ax2, line_only=False)
    # plt.show()
