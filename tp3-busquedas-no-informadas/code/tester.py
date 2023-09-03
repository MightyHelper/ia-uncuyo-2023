import pandas as pd

from grid_traversal_env import GridTraversalDiscreteEnvironment
from dfs_agent import DFSDiscreteAgent
from bfs_agent import BFSDiscreteAgent
from dijkstra_agent import DijkstraDiscreteAgent
from ldfs_agent import LDFSDiscreteAgent
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import tqdm

from lib.discrete_agent import DiscreteAgent
from lib.discrete_env import DiscreteEnvironment
from lib.random_discrete_agent import RandomAgent
# noinspection PyUnresolvedReferences
import lib.istarmap_pool  # Do not delete

agent_types = ['random', 'dfs', 'bfs', 'dijkstra', 'ldfs-00.5', 'ldfs-01.0', 'ldfs-02.0', 'ldfs-04.0', 'ldfs-16.0']


def simulate_agent_env(agent: DiscreteAgent, env: DiscreteEnvironment):
    observation = env.initial_state()
    while not env.is_done():
        action = agent.get_action(observation)
        observation = env.process_action(action)
    return env.get_performance(), env.get_restriction_stats()


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
    raise Exception(f"Unknown agent type: {agent_type}")


def gen_simulate_agent_env(agent_type: str, env: GridTraversalDiscreteEnvironment):
    agent = gen_agent(env, agent_type)
    performance, metadata = simulate_agent_env(agent, env)
    metadata = {**metadata, 'agent_type': agent_type, 'performance': performance, 'env': env.id}
    return metadata


def write_report(info_tables, data_to_plot, df, env_filenames):
    with open('../tp3-reporte.md', 'w') as f:
        f.write("# TP3 Report (B)\n")
        f.write("## Data plots\n");
        plot_results(data_to_plot, f, len(env_filenames))
        f.write("## Environments\n");
        plot_envs(env_filenames, f)
        f.write("## Tabular data\n");
        plot_md_tables(f, info_tables)
        f.write("## Data\n");
        plot_pd_tables(f, info_tables)
        f.write("## Raw Data\n");
        plot_csv(df, f)
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


def plot_csv(df, f):
    f.write(f"```csv\n{df.to_csv()}\n```\n")
    with open('../no-informada-results.csv', 'w') as f2:
        # Rename the index to 'run_n'
        df.index.name = 'run_n'
        df = df.rename(columns={'agent_type': 'algorithm_name', 'env': 'estate_n'})
        df['solution_found'] = df['performance'] == 1.0
        df = df.drop(columns=['performance', 'used_time'])
        df = df.sort_values(by=['algorithm_name', 'estate_n'])
        df = df.reset_index()
        df = df[['algorithm_name', 'run_n', 'estate_n', 'solution_found']]
        # Sort by algorithm name, then by estate_n
        f2.write(df.to_csv())


def plot_envs(env_filenames, f):
    f.write("Green = Start pos\n\n")
    f.write("Red = Target pos\n\n")
    for i, filename in enumerate(env_filenames):
        f.write(f"![Environment {i}]({filename[len('../'):]})\n")
        f.write("\n")


def plot_pd_tables(f, plots):
    for name, plot in plots.items():
        f.write(f"### {name}\n```pd\n{plot.to_string()}\n```\n")
    f.write("\n")


def plot_md_tables(f, plots):
    for name, plot in plots.items():
        f.write(f"### {name}\n{plot.to_markdown()}\n")


def plot_results(results, f, n_envs):
    global agent_types
    for (title, column, desc), plot in results.items():
        if title.startswith('Overall'):
            continue
        plot_results_by_env(agent_types, column, desc, f, n_envs, plot, title)
    for (title, column, desc), plot in results.items():
        if not title.startswith('Overall'):
            continue
        plot_results_overall(column, desc, f, plot, title)


def plot_results_overall(column, desc, f, plot, title):
    f.write(f"## {title} Plot\n")
    plot = plot[column]
    plot = plot.reset_index()
    plot = plot.set_index(['agent_type'])
    plot = plot.rename(columns={0: column})
    plot_title = f"{title} ({desc})"
    plot_and_out(f, title, f"o_Bar_{title}_log.png", plot.plot(kind='bar', title=plot_title))
    f.write("\n")


def plot_results_by_env(agent_types, column, desc, f, n_envs, plot, title):
    f.write(f"## {title} Plot\n")
    plot = plot[column]
    plot_title = f"{title} ({desc})"
    d1 = len(agent_types)
    d2 = n_envs // 4
    boxplot = plot.plot(kind='box', logy=True, title=plot_title, figsize=(d1, d2))
    bar_plots = plot.plot(kind='bar', legend=False, subplots=True, logy=True, title=plot_title, figsize=(d2, d1))
    plot_and_out(f, title, f"Box_{title}_log.png", boxplot)
    plot_and_out(f, title, f"Bar_{title}_log.png", bar_plots[0])
    f.write("\n")


def plot_and_out(file, index, file_name, df):
    fig = df.get_figure()
    fig.tight_layout()  # Avoid overlapping labels
    fig.autofmt_xdate(rotation=90)  # Rotate x labels 90 deg
    fig.savefig(f"../plots/{file_name}")
    file.write(f"![Plot {index}](plots/{file_name})\n")


def plot_env(env, i):
    filename = f'../plots/env_{i}.png'
    plt.clf()
    plt.imshow(env.environment, cmap='Greys', interpolation='nearest')
    plt.scatter(env.agent_pos[1], env.agent_pos[0], c='r', marker=',')
    plt.scatter(env.target_pos[1], env.target_pos[0], c='g', marker=',')
    plt.title(f'Environment {i}')
    plt.axis('off')
    plt.savefig(filename)
    return filename


def do_aggregation(dfg):
    return dfg.agg({'used_time': ['mean', 'std'], 'performance': ['mean', 'std']})


def analyse_results(env_filenames, results):
    print("Generating Reports", flush=True)
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
    data_to_plot = {
        ('Performance', 'performance', 'Higher is better'): df.groupby(['env', 'agent_type']).mean().unstack(
            ['agent_type']),
        ('Used time', 'used_time', 'Lower is better'): df.groupby(['env', 'agent_type']).mean().unstack(['agent_type']),
        ('Overall Performance', 'performance', 'Higher is better'): df.groupby(['agent_type']).mean().unstack(
            ['agent_type']),
        ('Overall Used time', 'used_time', 'Lower is better'): df.groupby(['agent_type']).mean().unstack(
            ['agent_type']),
    }
    write_report(info_tables, data_to_plot, df, env_filenames)


def test_agents(environments, n_envs):
    global agent_types
    print("Testing agents...", flush=True)
    with mp.Pool(n_envs) as pool:
        states = [(agent_type, env) for env in environments for agent_type in agent_types]
        results = [*tqdm.tqdm(pool.istarmap(gen_simulate_agent_env, states),
                              total=len(states))]  # results = pool.starmap(gen_simulate_agent_env, states)
    return results


def generate_environments(env_size, max_time, n_envs, wall_percent):
    print("Generating Environments", flush=True)
    with mp.Pool(n_envs) as pool:
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


def main():
    n_envs = 30
    env_size = 100  # n x n
    wall_percent = 0.08
    max_time = 10_000  # Max time to simulate if not solved
    env_filenames, environments = generate_environments(env_size, max_time, n_envs, wall_percent)
    results = test_agents(environments, n_envs)
    analyse_results(env_filenames, results)


if __name__ == "__main__":
    main()
