#!/usr/bin/python3
import random

from grid_traversal_env import GridTraversalDiscreteEnvironment
import pandas as pd
from dfs_agent import DFSDiscreteAgent
import sys
sys.setrecursionlimit(100*10*5)


from search_tester import SearchTester
import numpy as np

if __name__ == "__main__":
    np.random.seed(52)
    random.seed(0)
    df = SearchTester(debug=False, parallel=True, progress=True)([
        ('environment', ['search']),
        ('agent_type', ['random', 'bfs', 'dfs', 'dijkstra']),
        # ('agent_type', ['dijkstra']),
        ('env_size', [7]),
        ('wall_percent', [0.08]),
        ('n_iter', range(0, 30)),
        ('max_time', [100_000])
    ])
    # for i in range(50):
    #     print(i)
    #     env = GridTraversalDiscreteEnvironment(np.array([100, 100]), 0.08, 100_000)
    #     agent = DFSDiscreteAgent(env)
    #
    #     obs = env.initial_state()
    #     while not env.is_done():
    #         obs = env.process_action(agent.get_action(obs))
    df = df.convert_dtypes()
    df = df.drop(columns=['n_iter', 'max_time'])
    arr = ['env_size']
    arr2 = ['wall_percent']
    df[arr] = df[arr].astype(int) # cast types
    df[arr2] = df[arr2].astype(float)
    print(df.dtypes)
    df = df.groupby(['environment', 'agent_type', 'env_size', 'wall_percent']).mean().unstack('wall_percent', fill_value=-1)
    print(df.to_string())
