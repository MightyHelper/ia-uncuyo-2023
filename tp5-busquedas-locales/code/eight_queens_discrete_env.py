from typing import Any

import numpy as np
from numba import jit
from lib.discrete_env import DiscreteEnvironment


@jit(nopython=True, fastmath=True)
def is_attacking(ax, ay, bx, by) -> bool:
    """Checks if two queens are attacking each other (in 2 dimensions). Blazing fast version"""
    dx = ax - bx
    dy = ay - by
    return dx == 0 or dy == 0 or dx == dy or dx == -dy


class EightQueensEnvironment:
    def __init__(self, size):
        self.size = size

    def score_config(self, config: list):
        """Counts the number of pairs of queens that are attacking each other
        Shape: (1 dim)
        """
        return self.do_score_config(config, self.size[0])

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def build_directions(dimensions):
        out = []
        for i in [-1, 0, 1]:
            if dimensions == 1:
                out.append(np.array([i]))
                continue
            other = EightQueensEnvironment.build_directions(dimensions - 1)
            for o in other:
                out.append(np.array([i, *o]))
        return out

    @staticmethod
    def test_is_attacking(sim_size=8, do_print=True):
        for i in range(sim_size):
            for j in range(sim_size):
                mat: np.ndarray = np.zeros((sim_size, sim_size))
                for k in range(sim_size):
                    for l in range(sim_size):
                        mat[k][l] = is_attacking(i, j, k, l)
                # if do_print:
                #     for k in range(sim_size):
                #         for l in range(sim_size):
                #             if k == i and l == j:
                #                 print("\033[1;33;40mX \033[0m", end="")
                #                 continue
                #             if mat[k][l] == 1:
                #                 print("\033[1;31;40mX \033[0m", end="")
                #             else:
                #                 print("\033[1;32;40mX \033[0m", end="")
                #         print()
                #     print()

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def do_score_config(config: list, size: int) -> int:
        for x in config:
            if x < 0 or x >= size:
                return 9999999999
        count = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_attacking(i, config[i], j, config[j]):
                    count += 1
        return count
