from typing import Any

import numpy as np

from lib.discrete_env import DiscreteEnvironment


class EightQueensEnvironment:
    def __init__(self, size):
        self.size = size

    def score_config(self, config: np.ndarray):
        """Counts the number of pairs of queens that are attacking each other
        Shape: (n-1 dims)
        """
        if np.any(config < 0) or np.any(config >= self.size[0]):
            return 9999999999
        count = 0
        if len(config.shape) == 1:
            for i in range(config.shape[0]):
                q1 = np.array([i, config[i]])
                for j in range(i + 1, config.shape[0]):
                    q2 = np.array([j, config[j]])
                    if self.is_attacking(q1, q2):
                        # print(q1, q2)
                        count += 1
        else:
            raise NotImplementedError("Not implemented for more than 1 dimension")
        return count


    def is_attacking(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Checks if two queens are attacking each other (in n dimensions)"""
        # cos-1 [ (a. b) / (|a| |b|) ]
        if np.any(a == b):
            return True

        directions = self.build_directions(len(a))
        directions = np.array([d for d in directions if np.any(d != 0)])
        vec = b - a
        angles = np.dot(vec, directions.T) / (np.linalg.norm(vec) * np.linalg.norm(directions, axis=1))
        valid_angles = np.array([0, 1, -1])
        result = np.any(angles[:, None] == valid_angles, axis=1)
        return np.any(result)

    def test_is_attacking(self):
        for i in range(8):
            for j in range(8):
                mat = np.zeros((8, 8))
                for k in range(8):
                    for l in range(8):
                        mat[k][l] = self.is_attacking(np.array([i, j]), np.array([k, l]))
                # print(mat == 1)
                # \033[1;31;40mX\033[0m if 1
                # \033[1;32;40mX\033[0m if 0

                for k in range(8):
                    for l in range(8):
                        if k == i and l == j:
                            print("\033[1;33;40mX \033[0m", end="")
                            continue
                        if mat[k][l] == 1:
                            print("\033[1;31;40mX \033[0m", end="")
                        else:
                            print("\033[1;32;40mX \033[0m", end="")
                    print()
                print()

    def build_directions(self, dimensions):
        out = []
        for i in [-1, 0, 1]:
            if dimensions == 1:
                out.append(np.array([i]))
                continue
            other = self.build_directions(dimensions - 1)
            for o in other:
                out.append(np.array([i, *o]))
        return out
