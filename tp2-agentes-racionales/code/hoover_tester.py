from typing import cast

import numpy as np

from hoover_agent import ReflexiveHooverAgent
from hoover_env import HooverDiscreteEnvironment

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


class HooverTester(EnvTester):
    def gen_env(self, **kwargs):
        # env_size: int, dirt_percent: float = 0.5, max_time: int = 1000
        env_size = int(kwargs['env_size'])
        dirt_percent = float(kwargs['dirt_percent'])
        max_time = int(kwargs['max_time'])
        return HooverDiscreteEnvironment(np.array([1 << env_size, 1 << env_size]), dirt_percent, max_time)

    def gen_agent(self, env: DiscreteEnvironment, *args, **kwargs) -> DiscreteAgent:
        agent_type = kwargs['agent_type']
        if agent_type == 'random':
            return RandomAgent(env)
        elif agent_type == 'reflexive-hoover':
            return ReflexiveHooverAgent(cast(GridDiscreteEnvironment, env))
        raise Exception(f"Unknown agent type: {agent_type}")
