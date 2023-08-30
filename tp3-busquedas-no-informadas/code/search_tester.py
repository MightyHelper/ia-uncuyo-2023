from typing import cast

import numpy as np

from dfs_agent import DFSDiscreteAgent
from grid_traversal_env import GridTraversalDiscreteEnvironment

try:
    from lib.discrete_agent import DiscreteAgent
    from lib.discrete_env import DiscreteEnvironment
    from lib.grid_discrete_env import GridDiscreteEnvironment
    from lib.random_discrete_agent import RandomAgent
    from lib.runner import EnvTester
except ModuleNotFoundError as e:
    path = "PYTHONPATH"
    print(f"\x1b[31;1mCould not find python module '{e.name}'. Please add it to env variable {path}\x1b[0m")
    import os

    print(f"Current {path}: {os.getenv(path)}")
    exit(-1)


class SearchTester(EnvTester):
    def gen_env(self, **kwargs):
        # env_size: int, dirt_percent: float = 0.5, max_time: int = 1000
        env_size = int(kwargs['env_size'])
        wall_percent = float(kwargs['wall_percent'])
        max_time = int(kwargs['max_time'])
        return GridTraversalDiscreteEnvironment(np.array([1 << env_size, 1 << env_size]), wall_percent, max_time)

    def gen_agent(self, env: DiscreteEnvironment, *args, **kwargs) -> DiscreteAgent:
        agent_type = kwargs['agent_type']
        if agent_type == 'random':
            return RandomAgent(env)
        elif agent_type == 'dfs':
            return DFSDiscreteAgent(cast(GridTraversalDiscreteEnvironment, env))
        raise Exception(f"Unknown agent type: {agent_type}")
