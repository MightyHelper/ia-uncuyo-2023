import multiprocessing
import random
import tqdm
import time

from agents5 import HillClimbingAgent, SimulatedAnnealingAgent, GeneticAlgorithmAgent, EightQueensBaseAgent
from agents6 import BacktrackingAgent, ForwardCheckingAgent
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


def gen_agent(agent: tuple[str, dict[str, float]]) -> EightQueensBaseAgent:
    name, params = agent
    if name == 'hill_climbing':
        return HillClimbingAgent()
    elif name == 'simulated_annealing':
        return SimulatedAnnealingAgent(params['t'], params['d'])
    elif name == 'genetic_algorithm':
        ga = GeneticAlgorithmAgent(params['p_size'], params['gen'], params['mut'])
        ga.set_mut(params['mut_F'])
        ga.set_cross(params['cross_F'])
        ga.set_pop(params['pop_F'])
        return ga
    elif name == 'backtracking':
        return BacktrackingAgent()
    elif name == 'forward_checking':
        return ForwardCheckingAgent()
    raise Exception(f"Unknown agent type: {name}")


def test_agent(name: tuple[str, dict[str, float]], size: int):
    env = gen_env(size)
    agent = gen_agent(name)
    # print(agent.h_values)
    np.random.seed(random.randint(0, 100000))
    start_seconds = time.time()
    result, score, visited, h_values = agent.solve(env, 1)
    end_seconds = time.time()
    metadata = {
        'agent': name[0],
        'agent_params': name[1],
        'size': size,
        'score': score,
        'visited': visited,
        'seconds': end_seconds - start_seconds,
        'h_values': h_values,
        'result': result,
    }
    return metadata


def test_agent_n_times(name: tuple[str, dict[str, float]], size: int, n: int):
    print(f"Testing agent! {name} {size}", flush=True)
    results = [test_agent(name, size) for i in range(n)]
    for i, v in enumerate(results):
        v['run'] = i
    return results


def test_agent_v_envs(names: list[tuple[str, dict[str, float]]], sizes: list[int], n: int):
    print(f"Testing agents! {len(names) * len(sizes) * n}", flush=True)
    with multiprocessing.Pool() as pool:
        out = pool.starmap(test_agent_n_times, [(name, size, n) for size in sizes for name in names])
        # out = [test_agent_n_times(name, size, n) for size in sizes for name in names]
    return [item for sublist in out for item in sublist]


def do_starmap(params: dict[str, list[float]]) -> list[dict[str, float]]:
    """Recursively explode params"""
    keys = list(params.keys())
    values = list(params.values())
    first_key = keys[0]
    rest_keys = keys[1:]
    out = []
    if len(keys) == 1:
        for v in values[0]:
            out.append({first_key: v})
    else:
        for v in values[0]:
            for x in do_starmap({k: v for k, v in zip(rest_keys, values[1:])}):
                x[first_key] = v
                out.append(x)
    return out


def main(
        simulations_to_run: list[tuple[str, dict[str, list[float]]]], sizes: list[int],
        n_iter: int = 1,
        out_file: str = 'results_h.pkl'
):
    start_time = time.time()
    simulations = []
    for name, params in simulations_to_run:
        if len(params) == 0:
            simulations.append((name, {}))
        else:
            # Starmap
            simulations += [(name, x) for x in do_starmap(params)]
    print("\n".join([str(x) for x in simulations]))
    results = test_agent_v_envs(simulations, sizes, n_iter)
    df = pandas.DataFrame(results)
    df.to_pickle(out_file)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")

if __name__ == '__main__':
    main([
        ('hill_climbing', {}),
        ('simulated_annealing', {'t': [50, 100], 'd': [0.999, 0.99, 0.9, 0.5, 0.1]}),
        ('genetic_algorithm', {
            'p_size': [50, 1000],
            'gen': [100],
            'mut': [0.1, 0.5],
            'mut_F': [0, 1],
            'cross_F': [0, 1],
            'pop_F': [0, 1]
        }),
    ], [4, 8, 10, 12, 15, 16, 32, 64, 128], 100, 'results_h.pkl')
