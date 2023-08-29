#!/usr/bin/python3
from hoover_tester import HooverTester
import numpy as np

if __name__ == "__main__":
    df = HooverTester(parallel=True)([
        ('environment', ['hoover']),
        ('agent_type', ['random', 'reflexive-hoover']),
        ('env_size', range(0, 9)),
        ('dirt_percent', np.arange(0, 1, 0.1)),
        ('n_iter', range(0, 100)),
        ('max_time', [100])
    ])
    print(df.to_string())
