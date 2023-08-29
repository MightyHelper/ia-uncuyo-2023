#!/usr/bin/python3
import pandas as pd

from hoover_tester import HooverTester
import numpy as np

if __name__ == "__main__":
    df = HooverTester(debug=False, parallel=True, progress=True)([
        ('environment', ['hoover']),
        ('agent_type', ['random', 'reflexive-hoover']),
        ('env_size', range(0, 9)),
        ('dirt_percent', np.arange(0, 1, 0.1)),
        ('n_iter', range(0, 10)),
        ('max_time', [1000])
    ])
    df = df.convert_dtypes()
    df = df.drop(columns=['n_iter', 'max_time'])
    arr = ['env_size']
    arr2 = ['dirt_percent']
    df[arr] = df[arr].astype(int) # cast types
    df[arr2] = df[arr2].astype(float)
    print(df.dtypes)
    df = df.groupby(['environment', 'agent_type', 'env_size', 'dirt_percent']).mean().unstack('dirt_percent', fill_value=-1)
    print(df.to_string())
