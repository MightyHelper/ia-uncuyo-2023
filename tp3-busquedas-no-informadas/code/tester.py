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
    metadata = {**metadata, 'agent_type':agent_type, 'performance':performance, 'env':env.id}
    return metadata


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
        print(df.to_string())
        print(df.groupby(['agent_type']).mean().to_string())
        print(df.drop(columns=['agent_type']).groupby(['env']).mean().to_string())
        print(df.pivot_table(index=['env'], columns=['agent_type'], values=['performance']).to_string())
