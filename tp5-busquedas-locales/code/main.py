from agent import HillClimbingAgent
from eight_queens_discrete_env import EightQueensEnvironment
import numpy as np

env = EightQueensEnvironment(np.array([10, 10]))
agent = HillClimbingAgent()
# print(env.score_config(np.array([1, 3, 5, 7, 2, 0, 6, 4])))
solution = agent.solve(env, 4)
score = env.score_config(solution)
print(solution, score)
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
# data = agent.compute_n_directions(env, 5)
# print(data.shape)
