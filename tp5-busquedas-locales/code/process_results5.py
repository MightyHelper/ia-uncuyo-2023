import pandas as pd
from matplotlib import pyplot as plt
import os
import math

def map_func_ver(x):
    if x == 0.0: return 'basic'
    elif x == 1.0: return 'permut-aware'
    elif x == 2.0: return 'permut-double'
    elif math.isnan(x): return ''
    else:
        raise Exception(f"Unknown value {x}")

def map_float_stirng(x):
    if math.isnan(x): return ''
    return str(x)

def process_results(filename):
    df = pd.read_pickle(filename)
    df.drop(columns=['h_values'], inplace=True)
    agent_param_list = [] # ['t', 'd','p_size','pop_F','cross_F','mut_F','mut','gen' ]
    unmarshaled_columns = df['agent_params'].apply(lambda x: pd.Series(x))
    df = pd.concat([df, unmarshaled_columns], axis=1)
    df = df.drop(columns=['agent_params'])
    df['solved'] = df['score'] == 0
    df['solved'] = df['solved'].astype(int)
    # df['pop_F'] = df['pop_F'].map(map_func_ver)
    # df['cross_F'] = df['cross_F'].map(map_func_ver)
    # df['mut_F'] = df['mut_F'].map(map_func_ver)
    # df['d'] = df['d'].map(map_float_stirng)
    # df['t'] = df['t'].map(map_float_stirng)
    # df['mut'] = df['mut'].map(map_float_stirng)
    # df['gen'] = df['gen'].map(map_float_stirng)
    # df['p_size'] = df['p_size'].map(map_float_stirng)

    score_plot = df.groupby(['agent', *agent_param_list], dropna=False, sort=True, group_keys=False).agg({'solved': 'mean'}).plot(kind="barh", title="Solved % (Higher is better)", figsize=(20,20), xlim=(0,1))
    # Stop ticks rendered outside of the plot
    score_plot.get_figure().savefig('score.png', bbox_inches='tight', pad_inches=0)
    visited_plot = df.plot(kind="box", by=['agent', *agent_param_list], column=['visited'], logy=True, title="Visited states (Lower is better)", rot=90, figsize=(20,10))[0]
    visited_plot.get_figure().savefig('visited.png', bbox_inches='tight', pad_inches=0)
    exec_time_log_plot = df.plot(kind="box", by=['agent', *agent_param_list], column=['seconds'], logy=True, title="Logarithmic Execution time (Lower is better)", rot=90, figsize=(20,10))[0]
    exec_time_log_plot.get_figure().savefig('exec_time_log.png', bbox_inches='tight', pad_inches=0)
    exec_time_plot = df.plot(kind="box", by=['size'], column=['seconds'], logy=False, title="Execution time (Lower is better)", rot=90, figsize=(20,10))[0]
    exec_time_plot.get_figure().savefig('exec_time.png', bbox_inches='tight', pad_inches=0)

    df[['agent','result']].to_csv('results.csv', index=False)


if __name__ == '__main__':
    process_results('results_h.pkl')
