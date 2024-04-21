"""
Code for creating toy switching system with same api as gym.
"""
from typing import List, Optional, Callable
from abc import ABC
import numpy as np


class Condition(ABC):

    def __init__(self) -> None:
        pass

    def evaluate(self, x) -> bool:
        raise NotImplementedError("This is ABC")


class Null(Condition):

    def evaluate(self, x) -> bool:
        return False


class HyperPlane(Condition):
    """
    Evaluates if x.c > d
    """
    
    def __init__(self, c, d) -> None:
        super().__init__()
        self.c = c
        self.d = d
    
    def evaluate(self, x) -> bool:
        return self.c @ x - self.d > 0 


class LinearSystem:

    def __init__(self, A, B, condition: Optional[Condition]=None) -> None:
        self.A = A
        self.B = B
        self.dims = self.A.shape[0]
        self.condition = Null() if condition is None else condition
    
    def forward(self, x, u):
        return self.A @ x + self.B @ u

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
        self.x = x if x is not None else np.zeros((self.dims, 1))
        pass

    def forward(self, x, u):
        for linear in self.linear_systems:
            if linear.condition.evaluate(x):
                return linear.forward(x, u)
        return self.linear_systems[0].forward(x, u)

    def step(self, u):
        """
        Returns:
            obs, reward, terminated, truncated, info
        """
        self.x = self.forward(self.x, u)
        return self.x, self.reward(self.x), None, None, None

    def reset():
        pass

    def reward(self, x, reward_func: Optional[Callable] = None):
        if reward_func:
            return reward_func(x)
        return -1


if __name__ == "__main__":

    class Positive(Condition):
        def evaluate(self, x) -> bool:
            return x[0] > 0
        
    hyperplane = HyperPlane(np.array([1]), 1)
    
    env = SwitchSystem(
        linear_systems=[
            LinearSystem(np.array([[2]]), np.array([[1]])),
            LinearSystem(np.array([[1]]), np.array([[1]]), condition=hyperplane)
        ],
        x = np.array([.2])
    )

    for _ in range(10):
        res = env.step(u=np.array([0]))
        print(res)


    
