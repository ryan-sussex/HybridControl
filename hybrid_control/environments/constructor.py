"""
Code for creating toy switching system with same api as gym.
"""
from typing import List, Optional, Callable
from abc import ABC
import numpy as np


class LinearSystem:

    def __init__(self, A, B, w = None, b = None, u_max=None) -> None:
        self.A = A
        self.B = B
        self.dims = self.A.shape[0]
        if w is None:
            w = np.zeros(self.dims)
        if b is None:
            b = 0
        self.w = w
        self.b = b
        self.u_max = u_max if u_max is not None else np.inf

    def forward(self, x, u, sigma=.1):
        u[u > self.u_max] = self.u_max
        u[u < -self.u_max] = -self.u_max
        print(u)
        return self.A @ x + self.B @ u + np.random.normal(np.zeros(len(x)), scale=sigma)

    def step():
        pass

    def reset():
        pass

    def reward(x, reward_func: Optional[Callable] = None):
        if reward_func:
            return reward_func(x)
        return -1


class SwitchSystem:

    def __init__(
            self, 
            linear_systems: List[LinearSystem],
            x = None
    ) -> None:
        self.dims = linear_systems[0].dims
        self.linear_systems = linear_systems
        self.x = x if x is not None else np.zeros((self.dims,))
        pass

    def forward(self, x, u):
        likelihood = [linear.w @ x + linear.b for linear in self.linear_systems]
        mode = np.argmax(likelihood)
        return self.linear_systems[mode].forward(x, u)

    def step(self, u):
        """
        Returns:
            obs, reward, terminated, truncated, info
        """
        self.x = self.forward(self.x, u)
        return self.x, self.reward(self.x), None, None, None

    def reset(self, *args, **kwargs):
        self.x = np.zeros((self.dims,))
        return self.x, None

    def reward(self, x, reward_func: Optional[Callable] = None):
        if reward_func:
            return reward_func(x)
        return -1
    
    def close(self):
        return None
