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


def convert_to_servo(
        linear_controller: LinearController, x_ref
) -> LinearController:
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
    B_shape = linear_controller.B.shape

    bias = (linear_controller.A- np.eye(A_shape[0])) @ x_ref
    bias = bias[:, None]
    zeros = np.zeros(A_shape[1])[:, None]
    ones = np.array([[1]])

    A = np.block(
        [
            [linear_controller.A, bias],
            [zeros.T, ones]
        ]
    )
    B = np.block(
        [[linear_controller.B], [np.zeros((1, B_shape[1]))]]
    )
    Q = np.zeros(A.shape)
    Q[:linear_controller.Q.shape[0], :linear_controller.Q.shape[1]] = (
        linear_controller.Q
    )
    return LinearController(A, B, Q, linear_controller.R)


def calculate_gain(A, B, Q, R, S):
    return np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A


def backwards_riccati(A, B, Q, R, S):
    return (
        Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ B.T @ S @ A
    )


if __name__ == "__main__":

    A = np.array([[-.1, .1], [.2, .1]])
    B = np.array([[1, 2], [0.1, 3]])

    Q = np.eye(2) * 100
    R = np.eye(2) 

    x_0 = np.array([0, 6])
    x_ref = np.array([-30, 20])
   
   
   
    lc = LinearController(A, B, Q, R)
    lc = convert_to_servo(lc, x_ref)



    # Simulate system    
    x = x_0
    x_bar = np.r_[x - x_ref, 1] # internal coords
    traj = [x]
    for t in range(100):
        u = lc.finite_horizon(x_bar, t=t, T=100)
        x = A @ x + B @ u + np.random.normal([0, 0], scale=0.2)
        x_bar = np.r_[x - x_ref, 1] # translate to internal coords
        traj.append(x)
    
    X = np.column_stack(traj)


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
