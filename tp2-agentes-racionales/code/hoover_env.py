from typing import Any

import numpy as np

from lib.grid_discrete_env import GridDiscreteEnvironment
from lib.restriction import TimeRestriction


class HooverDiscreteEnvironment(GridDiscreteEnvironment):
    def __init__(self, dims: np.ndarray, dirt_probability: float = 0.5, max_time=1000):
        self.dirt_probability = dirt_probability
        super().__init__(dims)
        self.add_restriction(TimeRestriction(max_time))
        self.remaining_dirty = np.sum(self.environment)
        self.cleaned_dirty = 0

    def init_random_env(self):
        return np.random.random(self.dims) < self.dirt_probability

    def generate_actions(self, dims) -> list[str]:
        return ["noop", "clean"] + self.generate_move_actions(dims)

    def initial_state(self) -> tuple[bool, np.ndarray]:
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def _accept_action(self, action) -> None:
        match self.actions[action]:
            case "noop":
                pass
            case "clean":
                if self.environment[tuple(self.agent_pos)]:
                    self.cleaned_dirty += 1
                    self.remaining_dirty -= 1
                self.environment[tuple(self.agent_pos)] = False
            case _:
                self.agent_pos += self.action_to_direction(action)
                self.agent_pos = np.clip(self.agent_pos, 0, self.dims - 1)

    def get_performance(self):
        return 1 if self.remaining_dirty == 0 else self.cleaned_dirty / (self.cleaned_dirty + self.remaining_dirty)

    def print(self):
        GridDiscreteEnvironment.print(self)
        print("Remaining dirty: ", self.remaining_dirty)
        print("Successfully cleaned: ", self.cleaned_dirty)

    def get_state(self) -> Any:
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def objective_reached(self) -> bool:
        return self.remaining_dirty == 0
