import random

from agents5 import EightQueensBaseAgent
from eight_queens_discrete_env import EightQueensEnvironment, is_attacking
from numba import njit, int64
from numba.typed import List


class BacktrackingAgent(EightQueensBaseAgent):
    def __init__(self):
        super().__init__()
        self.env = None

    def solve(self, env: EightQueensEnvironment, lookahead: int = 1) -> tuple[list[int], int, int, list[int]]:
        self.env = env
        config = next(self.bt_solve_generator2([], self.env.size[0], set(range(self.env.size[0]))))
        return config, 0, self.total_visited, self.h_values

    def bt_solve_generator2(self, config: list[int], size: int, remaining: set[int]):
        for i in [*remaining]:
            config.append(i)
            self.total_visited += 1
            if EightQueensEnvironment.is_correct_no_clones(config, len(config)):
                remaining.remove(i)
                if len(config) == size:
                    yield config
                else:
                    yield from self.bt_solve_generator2(config, size, remaining)
                remaining.add(i)
            config.pop()


class ForwardCheckingAgent(EightQueensBaseAgent):
    def solve(self, env: EightQueensEnvironment, lookahead: int = 1) -> tuple[list[int], int, int, list[int]]:
        size = env.size[0]
        domain = [{*range(size)} for _ in range(size)]
        config = next(self.fc_solve_generator([], domain, size))
        return config, 0, self.total_visited, self.h_values

    def fc_solve_generator(self, config, domain, size):
        for i in [*domain[len(config)]]:
            config.append(i)
            old_domain = [{*d} for d in domain]
            self.total_visited += 1
            if ForwardCheckingAgent.update_domain(config, domain):
                if len(config) == size:
                    yield config
                else:
                    yield from self.fc_solve_generator(config, domain, size)
            domain = old_domain
            config.pop()

    @staticmethod
    def update_domain(config, domain):
        i_x = len(config) - 1
        i_y = config[-1]
        for j_x in range(i_x + 1, len(domain)):
            dx = i_x - j_x
            dom = domain[j_x]
            for j_y in [*dom]:
                if (dy := i_y - j_y) == 0 or dx == dy or dx == -dy: dom.remove(j_y)
            if not dom: return False
        return True
