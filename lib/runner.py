#!/usr/bin/python3
import multiprocessing
from abc import ABC, abstractmethod
from typing import Iterable, Any

import numpy as np
import pandas as pd

from lib.discrete_agent import DiscreteAgent
from lib.discrete_env import DiscreteEnvironment



class EnvTester(ABC):
    """
    Abstract class for testing environments and agents.
    """
    def __init__(self, debug=False, parallel=True):
        self.debug = debug
        self.parallel = parallel

    def simulate(self, agent: DiscreteAgent, environment: DiscreteEnvironment):
        if self.debug:
            environment.print()
            agent.print()
        state = environment.initial_state()
        while not environment.is_done():
            action = agent.get_action(state)
            if self.debug:
                print("-------------------------------------")
                environment.print()
                print("Action:", environment.actions[action])
                print("Performance:", environment.get_performance())
            state = environment.process_action(action)
        if self.debug:
            print("===================================")
            print("Failing restriction:", environment.find_failing_restriction())
            print("Performance:", environment.get_performance())
        return environment.get_performance()

    def eval_func(self, **kwargs):
        env = self.gen_env(**kwargs)
        agent = self.gen_agent(env, **kwargs)
        self.simulate(agent, env)
        return self.get_env_stats(env, **kwargs)

    def apply_eval_func(self, kwargs: dict[str, Any]):
        return self.eval_func(**kwargs)

    def dispatch_loops(self, **kwargs: Iterable):
        loops = [kwargs[k] for k in kwargs]
        cartesian_product = np.array(np.meshgrid(*loops)).T.reshape(-1, len(loops))
        return self.dispatch_parallel_loops(cartesian_product, kwargs) if self.parallel \
            else self.dispatch_serial_loops(cartesian_product, kwargs)

    def dispatch_serial_loops(self, cartesian_product, kwargs):
        cartesian_product = [dict(zip(kwargs.keys(), cartesian_product[i])) for i in range(len(cartesian_product))]
        print(f"Running {len(cartesian_product)} tests, using 1 cpu.")
        return [self.eval_func(**x) for x in cartesian_product]

    def dispatch_parallel_loops(self, cartesian_product, kwargs):
        cartesian_product = [[dict(zip(kwargs.keys(), cartesian_product[i]))] for i in range(len(cartesian_product))]
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            print(
                f"Running {len(cartesian_product)} tests, using {multiprocessing.cpu_count()} cpus = "
                f"{len(cartesian_product) / multiprocessing.cpu_count()} tests per cpu"
            )
            return pool.starmap(self.apply_eval_func, cartesian_product)

    def __call__(self, keys: list[tuple[str, Iterable]]) -> pd.DataFrame:
        kk = [k for k, _ in keys]
        result = self.dispatch_loops(**{k: v for k, v in keys})
        df = pd.DataFrame(result)
        df = df.sort_values(by=kk)
        df = df.reindex(columns=[*kk, 'performance', 'used_time', 'objective_reached'])
        df = df.reset_index()
        df = df.drop(columns=['index'])
        return df

    def get_env_stats(self, env: DiscreteEnvironment, **kwargs):
        out = {k: v for restriction in env.restrictions for k, v in restriction.get_stats().items()}
        out.update(kwargs)
        out['performance'] = env.get_performance()
        out['objective_reached'] = env.objective_reached()
        return out

    @abstractmethod
    def gen_env(self, **kwargs):
        pass

    @abstractmethod
    def gen_agent(self, env: DiscreteEnvironment, **kwargs):
        pass
