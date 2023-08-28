from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from lib.discrete_env import DiscreteEnvironment


class GridDiscreteEnv(DiscreteEnvironment, ABC):
    def __init__(self, dims: np.ndarray, max_time=1000):
        self.dims = np.array([5, 3]) if dims is None else dims  # Could be 1d, 2d, 3d, etc.
        super().__init__(self.generate_actions(self.dims))
        self.__actions_to_skip = len([i for i, a in enumerate(self.actions) if a.startswith("Move")])
        self.agent_pos = self.gen_random_pos()
        self.environment = self.init_random_env()
        self.remaining_time = max_time
        self.used_time = 0

    @abstractmethod
    def generate_actions(self, dims) -> list[str]:
        """Generate all non-move actions first"""
        pass

    def gen_random_pos(self):
        return np.floor(np.random.random(len(self.dims)) * self.dims).astype(dtype=int)

    def generate_move_actions(self, dims: np.ndarray) -> list[str]:
        actions = []
        for i in range(len(dims)):
            base = np.zeros(len(dims), dtype=int)
            base[i] = 1
            actions.append(f"Move {base}")
        for i in range(len(dims)):
            base = np.zeros(len(dims), dtype=int)
            base[i] = -1
            actions.append(f"Move {base}")
        return actions


    def init_random_env(self):
        return np.zeros_like(self.dims)

    def gen_random_pos(self):
        return np.floor(np.random.random(len(self.dims)) * self.dims).astype(dtype=int)

    def initial_state(self) -> tuple[Any, np.ndarray]:
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def action_to_direction(self, action):
        direction_code = action - 2
        direction = np.zeros(len(self.dims), dtype=int)
        direction[(direction_code % len(self.dims))] = 1 if direction_code < len(self.dims) else -1
        return direction

    def print(self):
        print("Agent position: ", self.agent_pos)
        print("Environment: \n", self.environment * 1)
