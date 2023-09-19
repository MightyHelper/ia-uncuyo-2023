from agent import HillClimbingAgent, SimulatedAnnealingAgent, GeneticAlgorithmAgent
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
from eight_queens_discrete_env import EightQueensEnvironment
import numpy as np

env = EightQueensEnvironment(np.array([15, 15]))
agent = HillClimbingAgent()
agent2 = SimulatedAnnealingAgent(100)
agent3 = GeneticAlgorithmAgent(1000, 100, 0.5)
print("Running", flush=True)
# print(env.score_config(np.array([1, 3, 5, 7, 2, 0, 6, 4])))
solution = agent.solve(env, 1)
solution2 = agent2.solve(env, 1)
solution3 = agent3.solve(env, 1)
score = env.score_config(solution)
score2 = env.score_config(solution2)
score3 = env.score_config(solution3)
print(solution, score)
print(solution2, score2)
print(solution3, score3)
