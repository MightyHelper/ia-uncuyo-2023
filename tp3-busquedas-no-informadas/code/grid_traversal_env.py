from typing import Any

import numpy as np
from lib.grid_discrete_env import GridDiscreteEnvironment
from lib.restriction import TimeRestriction, PathTrackingRestriction


class GridTraversalDiscreteEnvironment(GridDiscreteEnvironment):
    WALL = 1
    EMPTY = 0

    def __init__(self, dims: np.ndarray, wall_probability: float = 0.2, max_time=1000):
        self.wall_probability = wall_probability
        self.dims = dims
        super().__init__(dims)
        self.target_pos = self.gen_random_pos()
        self.add_restriction(TimeRestriction(max_time))
        self.add_restriction(PathTrackingRestriction())
        self.initial_distance = np.linalg.norm(self.target_pos - self.agent_pos)

    def init_random_env(self):
        return np.random.random(self.dims) < self.wall_probability

    def generate_actions(self, dims) -> list[str]:
        return ["halt"] + self.generate_move_actions(dims)

    def get_performance(self) -> float:
        # return self.target_pos.distance(self.agent_pos)
        if self.initial_distance == 0: return 1
        dist_between = np.linalg.norm(self.target_pos - self.agent_pos)
        if dist_between == 0: return 1
        return np.exp(-dist_between / self.initial_distance)

    def get_state(self) -> Any:
        return self.environment, self.agent_pos

    def _accept_action(self, action) -> None:
        direction = self.action_to_direction(action)
        new_pos = self.agent_pos + direction
        # self.disp()
        if self.is_valid_pos(new_pos):
            self.agent_pos = new_pos
        # else:
            # print("Invalid action", new_pos)

    def objective_reached(self) -> bool:
        # return self.agent_pos == self.target_pos
        # Use .all
        return bool(np.all(self.agent_pos == self.target_pos))

    def is_valid_pos(self, pos):
        return self._is_valid_pos(pos, self.environment)

    def _is_valid_pos(self, pos, env):
        if self.is_out_of_bounds(pos): return False
        return env[tuple(pos)] != self.WALL

    def is_out_of_bounds(self, pos):
        return np.any(pos < 0) or np.any(pos >= self.dims)

    def print(self):
        super().print()
        print("Target position: ", self.target_pos)
        self.disp()

    def disp(self):
        tmp = self.environment*1
        tmp[tuple(self.agent_pos)] = -1
        tmp[tuple(self.target_pos)] = -2
        print("Environment: \n")
        for row in tmp:
            for cell in row:
                if cell == self.WALL:
                    print("#", end="  ")
                elif cell == self.EMPTY:
                    print(" ", end="  ")
                elif cell == -1:
                    print("\x1b[1;32mA\x1b[0m", end="  ")
                elif cell == -2:
                    print("\x1b[1;31mT\x1b[0m", end="  ")
                else:
                    print("?", end="  ")
            print()


    def is_solvable(self, candidate_env: np.ndarray):
        """Check if the environment is solvable using bfs"""
        if candidate_env[tuple(self.agent_pos)] == self.WALL: return False
        if candidate_env[tuple(self.target_pos)] == self.WALL: return False
        visitable = np.zeros(self.dims, dtype=bool)
        verified = np.zeros(self.dims, dtype=bool)
        to_verify = [self.agent_pos]
        while len(to_verify) > 0:
            if len(to_verify) > 10000:
                print("Too many to verify")
            current = to_verify.pop(0)
            visitable[tuple(current)] = True
            for action in range(len(self.actions)):
                print("verifying", len(to_verify))
                direction = self.action_to_direction(action)
                new_pos = current + direction
                if self._is_valid_pos(new_pos, candidate_env) and not visitable[tuple(new_pos)]:
                    if not verified[tuple(new_pos)]:
                        verified[tuple(new_pos)] = True
                        to_verify.append(new_pos)
                    if np.all(new_pos == self.target_pos):
                        return True
        return False
