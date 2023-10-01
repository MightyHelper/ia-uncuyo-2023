import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def process_results(file):
    df_h = pd.read_pickle(file)
    # print(df_h[['agent', 'agent_params', 'size', 'h_values']])
    df_h['agent_params'] = df_h['agent_params'].map(
        lambda h_map: "\n".join([f"{k}:{v:8.3f}" for k, v in h_map.items()]))
    for agent in df_h['agent'].unique():
        agent_df = df_h[df_h['agent'] == agent]
        for agent_params in agent_df['agent_params'].unique():
            plt.clf()
            params_df = agent_df[agent_df['agent_params'] == agent_params]
            for index, row in params_df.iterrows():
                # colorize by size
                plt.plot(row['h_values'], label=f"{row['size']}", color=f"C{row['size']}")
            plt.title(f"{agent} agent\n{row['agent_params']}")
            # Expand the image to fit the legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.xlabel("Iteration")
            plt.ylabel("H value")
            plt.tight_layout()
            nl = '\n'
            filename = row['agent_params'].replace(nl, ' ').replace(' ', '_').replace(':', '_').replace('.', '_')
            # reduce _+ to _ using regexp
            filename = re.sub(r'_+', '_', filename)
            plt.savefig(f"h_{agent}_{filename}.png")
            print(f"![{agent} - {filename}](./code/h_{agent}_{filename}.png)")

if __name__ == '__main__':
    process_results('results_h.pkl')
