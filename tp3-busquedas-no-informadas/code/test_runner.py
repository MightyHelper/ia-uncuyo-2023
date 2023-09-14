import pandas as pd
from matplotlib import pyplot as plt

from grid_traversal_env import GridTraversalDiscreteEnvironment
from dfs_agent import DFSDiscreteAgent
from bfs_agent import BFSDiscreteAgent
from dijkstra_agent import DijkstraDiscreteAgent
from ldfs_agent import LDFSDiscreteAgent
from a_star_agent import AStarDiscreteAgent
import numpy as np
import multiprocessing as mp
import tqdm

from lib.discrete_agent import DiscreteAgent
from lib.random_discrete_agent import RandomAgent
# noinspection PyUnresolvedReferences
import lib.istarmap_pool  # Do not delete
from lib.utils import plot_results, do_aggregation, plot_env, plot_md_tables, plot_pd_tables, plot_envs, plot_csv, \
    simulate_agent_env


def gen_agent(env: GridTraversalDiscreteEnvironment, agent_type: str) -> DiscreteAgent:
    if agent_type == 'random':
        return RandomAgent(env)
    elif agent_type == 'dfs':
        return DFSDiscreteAgent(env)
    elif agent_type == 'bfs':
        return BFSDiscreteAgent(env)
    elif agent_type == 'dijkstra':
        return DijkstraDiscreteAgent(env)
    elif agent_type == 'ldfs-00.5':
        return LDFSDiscreteAgent(env, 0.5)
    elif agent_type == 'ldfs-01.0':
        return LDFSDiscreteAgent(env, 1.0)
    elif agent_type == 'ldfs-02.0':
        return LDFSDiscreteAgent(env, 2.0)
    elif agent_type == 'ldfs-04.0':
        return LDFSDiscreteAgent(env, 4.0)
    elif agent_type == 'ldfs-16.0':
        return LDFSDiscreteAgent(env, 16.0)
    elif agent_type == 'a*':
        return AStarDiscreteAgent(env)
    raise Exception(f"Unknown agent type: {agent_type}")


def gen_simulate_agent_env(agent_type: str, env: GridTraversalDiscreteEnvironment):
    agent = gen_agent(env, agent_type)
    performance, metadata = simulate_agent_env(agent, env)
    if agent_type == 'random':
        metadata['explored'] = metadata['used_time']
    metadata = {**metadata, 'agent_type': agent_type, 'performance': performance, 'env': env.id}
    return metadata


def analyse_results(env_filenames, results, report, agent_types, csv_out, tpn):
    print("Generating Reports", flush=True)
    for result in results:
        result['path'] = " ".join([str(x) for x in result['path']]) if 'path' in result else ""
    df = pd.DataFrame(results)
    df = df.sort_values(by=['agent_type', 'used_time', 'performance'])
    performance_of_agents_by_env = df.pivot_table(index=['env'], columns=['agent_type'],
                                                  values=['performance', 'used_time'])
    used_time_by_env = do_aggregation(df.drop(columns=['agent_type']).groupby(['env']))
    agent_used_time = do_aggregation(df.drop(columns=['env']).groupby(['agent_type']))
    info_tables = {
        'Performance of agents by environment': performance_of_agents_by_env,
        'Used time / performance by environment': used_time_by_env,
        'Used time / performance by agent': agent_used_time
    }
    path = df['path']
    df = df.drop(columns=['path'])
    # print(df.to_string())
    data_to_plot = {
        ('Performance', 'performance', 'Higher is better'): df.groupby(['env', 'agent_type']).mean().unstack(
            ['agent_type']),
        ('Used time', 'used_time', 'Lower is better'): df.groupby(['env', 'agent_type']).mean().unstack(['agent_type']),
        ('Explored', 'explored', 'Lower is better'): df.groupby(['env', 'agent_type']).mean().unstack(['agent_type']),
        ('Overall Performance', 'performance', 'Higher is better'): df.groupby(['agent_type']).mean().unstack(
            ['agent_type']),
        ('Overall Used time', 'used_time', 'Lower is better'): df.groupby(['agent_type']).mean().unstack(
            ['agent_type']),
        ('Overall Explored', 'explored', 'Lower is better'): df.groupby(['agent_type']).mean().unstack(['agent_type']),
    }
    write_report(info_tables, data_to_plot, df, env_filenames, report, agent_types, csv_out, tpn, path)


def test_agents(environments, n_envs, agent_types):
    print("Testing agents...", flush=True)
    with mp.Pool(n_envs) as pool:
        states = [(agent_type, env) for env in environments for agent_type in agent_types]
        results = [*tqdm.tqdm(pool.istarmap(gen_simulate_agent_env, states),
                              total=len(states))]  # results = pool.starmap(gen_simulate_agent_env, states)
    return results


def generate_environments(env_size, max_time, n_envs, wall_percent):
    print("Generating Environments", flush=True)
    with mp.Pool(8) as pool:
        states = [(env_size, i, max_time, wall_percent) for i in range(0, n_envs)]
        result = [*tqdm.tqdm(pool.istarmap(generate_env, states), total=len(states))]
        env_filenames = [x[0] for x in result]
        environments = [x[1] for x in result]
    return env_filenames, environments


def generate_env(env_size, i, max_time, wall_percent):
    import random
    np.random.seed(random.randint(0, 1000000))
    env = GridTraversalDiscreteEnvironment(np.array([env_size, env_size]), wall_percent, max_time)
    env.id = i
    return plot_env(env, i), env


def compact_path(path):
    split = path.split(" ")
    out = []
    # Compact a a a a a to 5xa if cout > 10
    count = 0
    last = None
    for x in split:
        if x == last:
            count += 1
        else:
            if last is not None:
                if count > 10:
                    out.append(f"{count}x{last}")
                else:
                    out.extend([last] * count)

            last = x
            count = 1
    if last is not None:
        out.append(f"{count}x{last}")
    return " ".join(out)


def plot_paths(path, df, f):
    f.write(f"Paths are defined as a sequence of actions, where each action is a number from 0 to 4, where 0 is halt, "
            f"1: +x, 2: -x, 3: +y, 4: -y.\n")
    for i, p in enumerate(path):
        f.write(f"### Path {i}, {df.iloc[i]['agent_type']} env {df.iloc[i]['env']}\n")
        f.write(f"{compact_path(p)}\n")
        f.write(f"Used time: {df.iloc[i]['used_time']}\n")
        f.write(f"Performance: {df.iloc[i]['performance']}\n")
        f.write(f"Explored: {df.iloc[i]['explored']}\n")
        f.write("\n")


def write_report(info_tables, data_to_plot, df, env_filenames, report, agent_types, csv_out, tpn, path):
    with open(report, 'w') as f:
        f.write(f"# TP{tpn} Report (B)\n")
        f.write("## Data plots\n")
        plot_results(data_to_plot, f, len(env_filenames), agent_types)
        f.write("## Environments\n")
        plot_envs(env_filenames, f)
        f.write("## Tabular data\n")
        plot_md_tables(f, info_tables)
        f.write("## Data\n")
        plot_pd_tables(f, info_tables)
        f.write("## Raw Data\n")
        plot_csv(df, f, csv_out)
        f.write("## Paths\n")
        plot_paths(path, df, f)
        if tpn != 3:
            return  # Only for tp3
        f.write("# TP3 Report (C)\n")
        # Cuál de los 3 algoritmos considera más adecuado para resolver el problema planteado en A)?. Justificar la respuesta.
        f.write("## Which algorithm is more suitable for the problem?\n")
        f.write(
            "The BFS algorithm is the most suitable for the problem because it is the one that finds a path with low effort.\n\n")
        f.write(
            "The DFS algorithm is the least suitable for the problem because it explores too much down a potentially incorrect path.\n\n")
        f.write(
            "The Dijkstra algorithm is the second most suitable for the problem because it is the one that finds the shortest path, but it uses more resources than BFS to do so (Because all edge weights are 1).\n\n")
        f.write(
            "The depth-bounded DFS is also bad because it might not even get to the solution if it's too far away\n\n")

        f.write("\n")


def main(report, csv_out, agent_types, tpn, n_envs=30, env_size=100, wall_percent=0.08, max_time=10_000):
    plt.tight_layout()
    env_filenames, environments = generate_environments(env_size, max_time, n_envs, wall_percent)
    results = test_agents(environments, n_envs, agent_types)
    analyse_results(env_filenames, results, report, agent_types, csv_out, tpn)


if __name__ == "__main__":
    main('../tp3-reporte.md', '../no-informada-results.csv',
         ['random', 'dfs', 'bfs', 'dijkstra', 'ldfs-00.5', 'ldfs-01.0', 'ldfs-02.0', 'ldfs-04.0', 'ldfs-16.0'], 3)
