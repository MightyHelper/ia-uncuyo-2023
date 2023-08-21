from lib.discrete_agent import DiscreteAgent
import numpy as np
from hoover_env import HooverDiscreteEnvironment


class ReflexiveHooverAgent(DiscreteAgent):
    def __init__(self, environment: HooverDiscreteEnvironment):
        super().__init__(environment)
        self.env_dims = environment.dims
        self.n_env_dims = len(self.env_dims)

    def get_action(self, observation: tuple[bool, np.ndarray]) -> int:
        is_dirty = observation[0]
        location = observation[1]
        if is_dirty:
            return 1  # CLEAN
        else:

            # v <
            # > ^
            # 00

            # v < < <
            # v > > ^
            # v ^ < <
            # > > > ^

            # v < < < < <
            # v > > > > ^
            # v ^ < < < <
            # v > > > > ^
            # v ^ < < < <
            # > > > > > ^
            x = location[0]
            y = location[1]

            # Left Column
            if x == 0:
                if y == 0:
                    return self.move(0)
                return self.move(1, True)

            # Right Column
            if x == self.env_dims[0] - 1:
                if y % 2 == 0:
                    return self.move(1)  # Up
                return self.move(0, True)  # Lef

            # Second Column
            if x == 1:
                if y % 2 == 0:
                    return self.move(0)
                if y == self.env_dims[1] - 1:
                    return self.move(0, True)
                return self.move(1)

            # Middle
            if y % 2 == 0:
                return self.move(0)
            return self.move(0, True)

    def print(self) -> None:
        action_map = ["N", "C", "v", ">", "^", "<"]
        # action_map = self.actions
        for i in range(self.env_dims[0]):
            for j in range(self.env_dims[1]):
                action = self.get_action((False, np.array([i, j])))
                # print(f"{i},{j}.{action}.{action_map[action]}", end=" ")
                # print(f"{i},{j}.{action_map[action]}", end=" ")
                print(f"{action_map[action]}", end=" ")
            print()

    def move(self, direction_code: int, reverse=False) -> int:
        if direction_code < 0: return self.move(-direction_code, reverse)
        return 2 + direction_code + (self.n_env_dims if reverse else 0)
