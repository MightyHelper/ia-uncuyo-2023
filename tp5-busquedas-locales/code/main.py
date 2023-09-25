import multiprocessing
import random
import tqdm
import time

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


def gen_agent(agent: tuple[str, dict[str, float]]) -> BaseLocalSearchAgent:
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
    raise Exception(f"Unknown agent type: {name}")


def test_agent(name: tuple[str, dict[str, float]], size: int):
    env = gen_env(size)
    agent = gen_agent(name)
    np.random.seed(random.randint(0, 100000))
    start_seconds = time.time()
    result, score, visited = agent.solve(env, 1)
    end_seconds = time.time()
    metadata = {
        'agent': name[0],
        'agent_params': name[1],
        'size': size,
        'score': score,
        'visited': visited,
        'seconds': end_seconds - start_seconds,
        'result': result,
    }
    return metadata


def test_agent_n_times(name: tuple[str, dict[str, float]], size: int, n: int, pool):
    # with multiprocessing.Pool() as pool:
    print(f"Testing agent! {name} {size}", flush=True)
    results = pool.starmap(test_agent, [(name, size)] * n)
    # results = list(tqdm.tqdm(pool.imap(test_agent, [(name, size)] * n), total=n))
    for i, v in enumerate(results):
        v['run'] = i
    return results


def test_agent_v_envs(names: list[tuple[str, dict[str, float]]], sizes: list[int], n: int):
    print(f"Testing agents! {len(names) * len(sizes) * n}", flush=True)
    with multiprocessing.Pool() as pool:
        out = [test_agent_n_times(name, size, n, pool) for size in sizes for name in names]
        # out = pool.starmap(test_agent_n_times, [(name, size, n) for size in sizes for name in names])
        return [item for sublist in out for item in sublist]


def do_starmap(params: dict[str, list[float]]) -> list[dict[str, float]]:
    """Recusively explode params"""
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


def main():
    start_time = time.time()
    simulations_to_run: list[tuple[str, dict[str, list[float]]]] = [
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
    ]
    # explode params
    simulations = []
    for name, params in simulations_to_run:
        if len(params) == 0:
            simulations.append((name, {}))
        else:
            # Starmap
            simulations += [(name, x) for x in do_starmap(params)]
    print("\n".join([str(x) for x in simulations]))
    n_iter = 128
    sizes = [4, 8, 10, 12, 15, 16, 32, 64, 128]
    results = test_agent_v_envs(
        simulations,
        sizes,
        n_iter
    )
    df = pandas.DataFrame(results)
    # print(results)
    # print(df.to_string())
    df.to_pickle('results2.pkl')
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")


def mini_map(h_map):
    return " ".join([f"{k}:{v:.3f}" for k, v in h_map.items()])


def solve1(df):
    """1. El número (porcentaje) de veces que se llega a un estado de solución óptimo."""
    # Use box plot, from 0 to 1, and save it to solved.png, key = [agent, agent_params], value = [solved]
    df.plot(by=['agent', 'agent_params'], column=['solved'], ylim=[0, 1], grid=True).get_figure().savefig(
        'solved.png')


def process_results():
    df = pandas.read_pickle('results1.pkl')
    # rename nanos to seconds
    df['seconds'] = df['nanos']  # Fix mistake
    df.drop(columns=['run', 'nanos'], inplace=True)
    df['agent_params'] = df['agent_params'].map(mini_map)
    df['solved'] = df['score'] == 0
    df['solved'] = df['solved'].astype(int)

    # 2. El tiempo de ejecución promedio y la desviación estándar para encontrar dicha solución. (se puede usar la función time.time() de python)
    # 3. La cantidad de estados previos promedio y su desviación estándar por los que tuvo que pasar para llegar a una solución.
    # 4. Generar un tabla con los resultados para cada uno de los algoritmos desarrollados y guardarla en formato .csv (comma separated value)
    # 5. Realizar un gráfico de cajas (boxplot) que muestre la distribución de los tiempos de ejecución de cada algoritmo. (ver gráfico de ejemplo)
    # B) Para cada uno de los algoritmos, graficar la variación  de la función h() a lo largo de las iteraciones. (Considerar solo una ejecución en particular)
    # C) Indicar según su criterio, cuál de los tres algoritmos implementados resulta más adecuado para la solución del problema de las n-reinas. Justificar.
    # Solve 1
    solve1(df)
    # print(df.to_string())
    # print(df2.to_string())


main()

# process_results()
