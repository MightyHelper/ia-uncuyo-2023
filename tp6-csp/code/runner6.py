from numba import jit, config

# config.DISABLE_JIT = True
from runner5 import *
import numpy as np
import pandas as pd
import eight_queens_discrete_env
from agents6 import BacktrackingAgent, ForwardCheckingAgent
import timeit

# main([
#     ('backtracking', {}),
# ], [18], 1, 'backtracking-results.pkl')

def main():
    ForwardCheckingAgent().solve(eight_queens_discrete_env.EightQueensEnvironment([18, 18]), 1)

def main2():
    BacktrackingAgent().solve(eight_queens_discrete_env.EightQueensEnvironment([18, 18]), 1)

print(pd.DataFrame(np.array(timeit.repeat(main, repeat=10, number=1))).agg(['mean', 'std', 'min', 'max']).T)
# print(pd.DataFrame(np.array(timeit.repeat(main2, repeat=10, number=1))).agg(['mean', 'std', 'min', 'max']).T)
# main()
# eight_queens_discrete_env.EightQueensEnvironment.test_is_attacking(4, True)
#  0  0.061142  0.000546  0.059168  0.063592
# 0  0.272878  0.00202  0.268591  0.278962
