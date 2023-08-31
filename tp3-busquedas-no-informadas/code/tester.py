import pandas as pd

from grid_traversal_env import GridTraversalDiscreteEnvironment
from dfs_agent import DFSDiscreteAgent
from bfs_agent import BFSDiscreteAgent
from dijkstra_agent import DijkstraDiscreteAgent
import numpy as np
import multiprocessing as mp

from lib.discrete_agent import DiscreteAgent
from lib.discrete_env import DiscreteEnvironment
from lib.random_discrete_agent import RandomAgent

import matplotlib as mpl


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
    raise Exception(f"Unknown agent type: {agent_type}")


def gen_simulate_agent_env(agent_type: str, env: GridTraversalDiscreteEnvironment):
    agent = gen_agent(env, agent_type)
    performance, metadata = simulate_agent_env(agent, env)
    metadata = {**metadata, 'agent_type': agent_type, 'performance': performance, 'env': env.id}
    return metadata


def write_report():
    global i, plot
    with open('../tp3-reporte.md', 'w') as f:
        f.write("# TP3 Report\n")
        f.write("## Data plots\n")
        for i, plot in enumerate(deagg_plots):
            f.write(f"## Plot {i}\n")
            plot_and_out(f, i, f"{i}_0.png", plot.plot(kind='box', title='Natrual'))
            plot_and_out(f, i, f"{i}_1.png", plot.plot(kind='box', logy=True, title='Logarithmic'))
            plot_and_out2(f, i, f"{i}_3.png", plot.plot(kind='bar', legend=False, subplots=True, title=agent_types))
            plot_and_out2(f, i, f"{i}_4.png", plot.plot(kind='bar', legend=False, subplots=True, logy=True, title=agent_types))
            f.write("\n")

        f.write("## Tabular data\n")
        for name, plot in plots.items():
            f.write(f"## {name}\n")
            f.write(plot.to_markdown())
            f.write("\n")
        f.write("## Data\n")
        for name, plot in plots.items():
            f.write(f"## {name}\n")
            f.write("```pd\n")
            f.write(plot.to_string())
            f.write("\n```\n")
        f.write("## Raw Data\n")
        f.write("```csv\n")
        f.write(df.to_csv())
        f.write("\n```")
        f.write("\n")


def plot_and_out(file, index, file_name, df):
    df.get_figure().savefig(f"../plots/{file_name}")
    file.write(f"![Plot {index}](plots/{file_name})\n")


def plot_and_out2(file, index, file_name, df):
    for i, x in enumerate(df):
        plot_and_out(file, f"{index}_{i}", f"{i}_{file_name}", x)
        return # Only one plot


if __name__ == "__main__":
    environments = []
    n_envs = 32
    env_size = 100
    wall_percent = 0.08
    max_time = 100_00
    for i in range(0, n_envs):
        env = GridTraversalDiscreteEnvironment(np.array([env_size, env_size]), wall_percent, max_time)
        env.id = i
        environments.append(env)
    # for environment in environments:
    #     environment.disp()
    print("Generated", flush=True)
    agent_types = ['random', 'dfs', 'bfs', 'dijkstra']
    with mp.Pool(n_envs) as pool:
        states = [(agent_type, env) for env in environments for agent_type in agent_types]
        print("Running", flush=True)
        results = pool.starmap(gen_simulate_agent_env, states)
        df = pd.DataFrame(results)
        df = df.sort_values(by=['agent_type', 'used_time', 'performance'])
        # print(df.to_string())
        # print(df.drop(columns=['env']).groupby(['agent_type']).mean().to_string())
        # print(df.drop(columns=['agent_type']).groupby(['env']).mean().to_string())
        performance_of_agents_by_env = df.pivot_table(index=['env'], columns=['agent_type'],
                                                      values=['performance', 'used_time'])
        used_time_by_env = df.drop(columns=['agent_type']).groupby(['env']).agg(
            {'used_time': ['mean', 'std'], 'performance': ['mean', 'std']})
        agent_used_time = df.drop(columns=['env']).groupby(['agent_type']).agg(
            {'used_time': ['mean', 'std'], 'performance': ['mean', 'std']})

        plots = {
            'Performance of agents by environment': performance_of_agents_by_env,
            'Used time / performance by environment': used_time_by_env,
            'Used time / performance by agent': agent_used_time
        }

        deagg_plots = [
            df.pivot_table(index=['env'], columns=['agent_type'], values=['performance']),
            df.pivot_table(index=['env'], columns=['agent_type'], values=['used_time']),
        ]

        write_report()
