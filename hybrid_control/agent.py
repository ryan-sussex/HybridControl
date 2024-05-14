from typing import Optional
import numpy as np
from pymdp.agent import Agent as _Agent


class Agent(_Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qs_prev: Optional[np.ndarray] = None
        self.qB: Optional[np.ndarray] = None
        self.mode_action_names = []