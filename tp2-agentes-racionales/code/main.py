#!/usr/bin/python3
from typing import Iterable, Any
import multiprocessing

try:
    from lib.random_discrete_agent import RandomAgent
except ModuleNotFoundError as e:
    path = "PYTHONPATH"
    print(f"\x1b[31;1mCould not find python module '{e.name}'. Please add it to env variable {path}\x1b[0m")
    import os

    print(f"Current {path}: {os.getenv(path)}")
    exit(-1)
import pandas as pd
import numpy as np
from hoover_agent import ReflexiveHooverAgent
from hoover_env import HooverDiscreteEnvironment

DEBUG = False


def gen_env(env_size, dirt_percent=0.5, max_time=1000):
    return HooverDiscreteEnvironment(np.array([1 << env_size, 1 << env_size]), dirt_percent, max_time)


def simulate(agent: ReflexiveHooverAgent, environment: HooverDiscreteEnvironment):
    global DEBUG
    if DEBUG: environment.print()
    if DEBUG: agent.print()
    state = environment.initial_state()
    while environment.remaining_time > 0 and environment.remaining_dirty > 0:
        action = agent.get_action(state)
        if DEBUG: print("-------------------------------------")
        if DEBUG: environment.print()
        if DEBUG: print("Action:", environment.actions[action])
        if DEBUG: print("Performance:", environment.get_performance())
        state = environment.accept_action(action)
    if DEBUG: print("===================================")
    if environment.remaining_time <= 0:
        if DEBUG: print("Simulation time exceeded")
    else:
        if DEBUG: print(f"Cleaned in {environment.used_time} steps!!")
    if DEBUG: print("Performance:", environment.get_performance())
    return environment.get_performance()


def test_agent(agent_type, env_size, env_dirt, max_time=1000):
    env = gen_env(env_size, env_dirt, max_time)
    agent = RandomAgent(env) if agent_type == 'random' else ReflexiveHooverAgent(env)
    simulate(agent, env)
    return env.used_time, env.get_performance()


def dispatch_loops(loops: list[Iterable], func, current_args: list[Any]):
    if len(loops) == 0: return func(*current_args)
    current_loop, other_loops = loops[0], loops[1:]
    return [dispatch_loops(other_loops, func, current_args + [i]) for i in current_loop]




def dispatch_loops_multithread(loops: list[Iterable], func: Any):
    cartesian_product = np.array(np.meshgrid(*loops)).T.reshape(-1, len(loops))
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool() as pool:
        # Get thread count
        print(f"Running {len(cartesian_product)} tests, using {multiprocessing.cpu_count()} cpus = {len(cartesian_product) / multiprocessing.cpu_count()} tests per cpu")
        # Run the function in parallel
        return pool.starmap(func, cartesian_product)

def test(agent_type: str, size: int, dirt: float, run: int, max_time: int):
    size = int(size)
    dirt = float(dirt)
    run = int(run)
    max_time = int(max_time)
    used_time, performance = test_agent(agent_type, size, dirt, max_time)
    return {
        'agent': agent_type,
        'size_value': size,
        'size': f"{1 << size}x{1 << size}",
        'cells': (1 << size) * (1 << size),
        'run': run,
        'dirt': dirt,
        'time': used_time,
        'performance': performance
    }


def benchmark_agents():
    print("Running benchmark")
    max_time = 1000
    iter_count = 1000
    loops = [
        ['random', 'reflexive'], # Agent_type
        range(0, 8),             # Size
        np.arange(0., 1., 0.01),        # [0.1, 0.2, 0.4, 0.8],    # Dirt %
        range(0, iter_count),    # Iter count
        [max_time]               # Max time
    ]


    output = dispatch_loops_multithread(loops, test)
    output = pd.DataFrame(output)
    output = output.drop(columns=['run', 'cells'])
    # output = output.groupby(['agent', 'size_value', 'size', 'cells', 'dirt', 'run']).mean()
    output = output.groupby(['agent', 'size', 'dirt']).mean()
    # remove the 'run' column
    print(output.to_string())  # Average over run data
    output.to_csv("output.csv")
    print("\n\n")
    # Ungroup
    output = output.reset_index()
    # Sort by agent, size, dirt
    output = output.sort_values(by=['agent', 'size_value', 'dirt'])
    output = output.drop(columns=['size_value'])
    # Export as md
    print(output.to_markdown())


if __name__ == "__main__":
    benchmark_agents()
