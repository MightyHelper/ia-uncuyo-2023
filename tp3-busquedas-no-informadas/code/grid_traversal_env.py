import numpy as np
from lib.grid_discrete_env import GridDiscreteEnv


class GridTraversalDiscreteEnv(GridDiscreteEnv):
    def generate_actions(self, dims) -> list[str]:
        return self.generate_move_actions(dims)

    def accept_action(self, action) -> tuple:
        if self.remaining_time <= 0:
            raise Exception("Simulation time exceeded")
        self.remaining_time -= 1
        self.used_time += 1
        self.agent_pos += self.action_to_direction(action)
        self.agent_pos = np.clip(self.agent_pos, 0, self.dims - 1)
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def get_performance(self) -> float:
        return self.environment[tuple(self.agent_pos)] == "G"

    def __init__(self, dims, wall_probability=0.5, max_time=1000):
        self.wall_probability = wall_probability
        super().__init__(dims, max_time)

    def init_random_env(self):
        return np.random.random(self.dims) < self.wall_probability
