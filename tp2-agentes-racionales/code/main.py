from hoover_agent import ReflexiveHooverAgent
from hoover_env import HooverDiscreteEnvironment
from lib.random_discrete_agent import RandomAgent
import pandas as pd
import numpy as np

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


def benchmark_agents():
    print("Running benchmark")
    output = pd.DataFrame(columns=['agent', 'size_value', 'size', 'cells', 'dirt', 'run', 'time', 'performance'])
    max_time = 1000
    for agent_type in ['random', 'reflexive']:
        for size in range(0, 8):
            for dirt in [0.1, 0.2, 0.4, 0.8]:
                for run in range(0, 10):
                    used_time, performance = test_agent(agent_type, size, dirt, max_time)
                    output = output._append({
                        'agent': agent_type,
                        'size_value': size,
                        'size': f"{1 << size}x{1 << size}",
                        'cells': (1 << size) * (1 << size),
                        'run': run,
                        'dirt': dirt,
                        'time': used_time,
                        'performance': performance
                    }, ignore_index=True)
                    # print(f"Agent: {agent_type} Size: {1 << size}x{1 << size}({(1 << size) * (1<<size)}), Dirt: {dirt}, Time Used: {used_time}, Performance: {performance}")
    # print(output.to_string()) # All data
    output = output.groupby(['agent', 'size_value', 'size', 'cells', 'dirt', 'run']).mean()
    # print(output.to_string()) # Grouped data
    # Average over 'run' column
    output = output.groupby(['agent', 'size_value', 'size', 'cells', 'dirt']).mean()
    print(output.to_string()) # Average over run data


if __name__ == "__main__":
    benchmark_agents()
