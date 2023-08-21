import numpy as np
from lib.discrete_env import DiscreteEnvironment


class HooverDiscreteEnvironment(DiscreteEnvironment):

    def __init__(self, dims: np.ndarray, dirt_probability: float = 0.5, max_time=1000):
        self.dims = np.array([5, 3]) if dims is None else dims  # Could be 1d, 2d, 3d, etc.
        super().__init__(HooverDiscreteEnvironment.__generate_actions(self.dims))
        self.agent_pos = self.gen_random_pos()
        self.environment = self.init_random_env(dirt_probability)
        self.remaining_time = max_time
        self.used_time = 0
        self.remaining_dirty = np.sum(self.environment)
        self.cleaned_dirty = 0

    def init_random_env(self, dirt_probability):
        return np.random.random(self.dims) < dirt_probability

    def gen_random_pos(self):
        return np.floor(np.random.random(len(self.dims)) * self.dims).astype(
            dtype=int)

    @staticmethod
    def __generate_actions(dims) -> list[str]:
        actions = ["noop", "clean"]
        for i in range(len(dims)):
            base = np.zeros(len(dims), dtype=int)
            base[i] = 1
            actions.append(f"Move {base}")
        for i in range(len(dims)):
            base = np.zeros(len(dims), dtype=int)
            base[i] = -1
            actions.append(f"Move {base}")
        return actions

    def initial_state(self) -> tuple[bool, np.ndarray]:
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def accept_action(self, action) -> tuple[bool, np.ndarray]:
        if self.remaining_time <= 0:
            raise Exception("Simulation time exceeded")
        self.remaining_time -= 1
        self.used_time += 1
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
        return self.environment[tuple(self.agent_pos)], self.agent_pos

    def action_to_direction(self, action):
        direction_code = action - 2
        direction = np.zeros(len(self.dims), dtype=int)
        direction[(direction_code % len(self.dims))] = 1 if direction_code < len(self.dims) else -1
        return direction

    def get_performance(self):
        if self.remaining_dirty == 0:
            return 1
        return self.cleaned_dirty / (self.cleaned_dirty + self.remaining_dirty)

    def print(self):
        print("Agent position: ", self.agent_pos)
        print("Environment: \n", self.environment * 1)
        print("Remaining time: ", self.remaining_time)
        print("Remaining dirty: ", self.remaining_dirty)
        print("Successfully cleaned: ", self.cleaned_dirty)
