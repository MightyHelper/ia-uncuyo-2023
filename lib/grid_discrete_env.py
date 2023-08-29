from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from lib.discrete_env import DiscreteEnvironment


class GridDiscreteEnvironment(DiscreteEnvironment, ABC):
    def __init__(self, dims: np.ndarray):
        self.dims = np.array([5, 3]) if dims is None else dims  # Could be 1d, 2d, 3d, etc.
        super().__init__(self.generate_actions(self.dims))
        self.__non_move_actions = len(self.actions) - (len(self.dims) << 1)
        self.agent_pos = self.gen_random_pos()
        self.environment = self.init_random_env()

    def generate_actions(self, dims) -> list[str]:
        return self.generate_move_actions(dims)

    def gen_random_pos(self):
        """Generate a random position in the environment"""
        return np.floor(np.random.random(len(self.dims)) * self.dims).astype(dtype=int)

    @staticmethod
    def generate_move_actions(dims: np.ndarray) -> list[str]:
        actions = []
        base = np.zeros(len(dims), dtype=int)
        for i in range(len(dims)):
            base[i] = +1; actions.append(f"Move {base}")
            base[i] = -1; actions.append(f"Move {base}")
            base[i] = 00
        return actions

    def init_random_env(self):
        return np.zeros_like(self.dims)

    def initial_state(self) -> tuple[Any, np.ndarray]:
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def action_to_direction(self, action):
        direction_code = action - self.__non_move_actions
        direction = np.zeros(len(self.dims), dtype=int)
        direction[(direction_code % len(self.dims))] = 1 if direction_code < len(self.dims) else -1
        return direction

    def print(self):
        super().print()
        print("Agent position: ", self.agent_pos)
        print("Environment: \n", self.environment * 1)
