from numba import jit, config

# config.DISABLE_JIT = True
import numpy as np
import pandas as pd
import eight_queens_discrete_env
from agents6 import BacktrackingAgent, ForwardCheckingAgent
import timeit


def run_agents():
    from runner5 import main
    main([
        ('backtracking', {}),
        ('forward_checking', {}),
    ], [4, 8, 10, 12, 15, 16, 18], 128, 'backtracking-results.pkl')


def process_results():
    from process_results5 import process_results
    process_results('backtracking-results.pkl')


def benchmark():
    n = 30
    r = 1

    def main():
        ForwardCheckingAgent().solve(eight_queens_discrete_env.EightQueensEnvironment([n, n]), 1)

    def main2():
        BacktrackingAgent().solve(eight_queens_discrete_env.EightQueensEnvironment([n, n]), 1)
    print(pd.DataFrame(np.array(timeit.repeat(main, repeat=r, number=1))).agg(['mean', 'std', 'min', 'max']).T)
    # print(pd.DataFrame(np.array(timeit.repeat(main2, repeat=r, number=1))).agg(['mean', 'std', 'min', 'max']).T)


if __name__ == '__main__':
    run_agents()
    process_results()
    # benchmark()
