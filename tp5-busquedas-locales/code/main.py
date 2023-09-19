import multiprocessing
import random

from agent import HillClimbingAgent, SimulatedAnnealingAgent, GeneticAlgorithmAgent, BaseLocalSearchAgent
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
from eight_queens_discrete_env import EightQueensEnvironment
import numpy as np
import warnings
import pandas

# Disable pesky numba warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


def gen_env(size: int) -> EightQueensEnvironment:
    return EightQueensEnvironment(np.array([size, size]))


def gen_agent(name: str) -> BaseLocalSearchAgent:
    if name == 'hill_climbing':
        return HillClimbingAgent()
    elif name == 'simulated_annealing-t-100-d-0.999':
        return SimulatedAnnealingAgent(100, 0.999)
    elif name == 'simulated_annealing-t-100-d-0.99':
        return SimulatedAnnealingAgent(100, 0.99)
    elif name == 'simulated_annealing-t-100-d-0.9':
        return SimulatedAnnealingAgent(100, 0.9)
    elif name == 'genetic_algorithm-mut-0.1':
        return GeneticAlgorithmAgent(1000, 100, 0.1)
    elif name == 'genetic_algorithm-mut-0.5':
        return GeneticAlgorithmAgent(1000, 100, 0.5)
    raise Exception(f"Unknown agent type: {name}")


def test_agent(name: str, size: int):
    env = gen_env(size)
    agent = gen_agent(name)
    np.random.seed(random.randint(0, 100000))
    start_nanos = np.datetime64('now')
    result, score, visited = agent.solve(env, 1)
    end_nanos = np.datetime64('now')
    metadata = {
        'agent': name,
        'size': size,
        'score': score,
        'visited': visited,
        'nanos': (end_nanos - start_nanos).astype(int),
        # 'result': result,
    }
    return metadata


def test_agent_n_times(name: str, size: int, n: int):
    # return [test_agent(name, size) for _ in range(n)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(test_agent, [(name, size)] * n)
        for i, v in enumerate(results):
            v['run'] = i
        return results


def test_agent_v_envs(names: list[str], sizes: list[int], n: int):
    out = [test_agent_n_times(name, size, n) for size in sizes for name in names]
    return [item for sublist in out for item in sublist]


def main():
    results = test_agent_v_envs(
        [
            'hill_climbing',
            'simulated_annealing-t-100-d-0.999',
            'simulated_annealing-t-100-d-0.99',
            'simulated_annealing-t-100-d-0.9',
            'genetic_algorithm-mut-0.1',
            'genetic_algorithm-mut-0.5'
        ],
        [4, 8, 16, 32, 64, 128],
        100
    )
    df = pandas.DataFrame(results)
    # print(results)
    # print(df.to_string())
    df.to_pickle('results.pkl')


def process_results():
    df = pandas.read_pickle('results.pkl')
    df.drop(columns=['run'], inplace=True)
    df2 = df.groupby(['agent', 'size']).agg({'score': ['mean', 'std'], 'visited': ['mean', 'std'], 'nanos': ['mean', 'std']})
    print(df.to_string())
    print(df2.to_string())


# main()

process_results()
