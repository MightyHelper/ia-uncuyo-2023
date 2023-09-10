from matplotlib import pyplot as plt

from lib.discrete_agent import DiscreteAgent
from lib.discrete_env import DiscreteEnvironment


def plot_results_by_env(agent_types, column, desc, f, n_envs, plot, title):
    f.write(f"## {title} Plot\n")
    plot = plot[column]
    plot_title = f"{title} ({desc})"
    d1 = len(agent_types)
    d2 = n_envs // 4
    boxplot = plot.plot(kind='box', logy=True, title=plot_title, figsize=(d1, d2))
    bar_plots = plot.plot(kind='bar', legend=False, subplots=True, logy=True, title=plot_title, figsize=(d2, d1))
    plot_and_out(f, title, f"Box_{title}_log.png", boxplot)
    plot_and_out(f, title, f"Bar_{title}_log.png", bar_plots[0])
    f.write("\n")


def plot_and_out(file, index, file_name, df):
    fig = df.get_figure()
    fig.tight_layout()  # Avoid overlapping labels
    fig.autofmt_xdate(rotation=90)  # Rotate x labels 90 deg
    fig.savefig(f"../plots/{file_name}")
    file.write(f"![Plot {index}](plots/{file_name})\n")


def plot_results_overall(column, desc, f, plot, title):
    f.write(f"## {title} Plot\n")
    plot = plot[column]
    plot = plot.reset_index()
    plot = plot.set_index(['agent_type'])
    plot = plot.rename(columns={0: column})
    plot_title = f"{title} ({desc})"
    plot_and_out(f, title, f"o_Bar_{title}.png", plot.plot(kind='bar', title=plot_title))
    plot_and_out(f, title, f"o_Bar_{title}_log.png", plot.plot(kind='bar', logy=True, title=plot_title))
    f.write("\n")


def plot_results(results, f, n_envs, agent_types):
    for (title, column, desc), plot in results.items():
        if title.startswith('Overall'):
            continue
        plot_results_by_env(agent_types, column, desc, f, n_envs, plot, title)
    for (title, column, desc), plot in results.items():
        if not title.startswith('Overall'):
            continue
        plot_results_overall(column, desc, f, plot, title)


def do_aggregation(dfg):
    return dfg.agg({'used_time': ['mean', 'std'], 'performance': ['mean', 'std']})


def plot_env(env, i):
    filename = f'../plots/env_{i}.png'
    plt.clf()
    plt.imshow(env.environment, cmap='Greys', interpolation='nearest')
    plt.scatter(env.agent_pos[1], env.agent_pos[0], c='r', marker=',')
    plt.scatter(env.target_pos[1], env.target_pos[0], c='g', marker=',')
    plt.title(f'Environment {i}')
    plt.axis('off')
    # smaller white border
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    return filename


def plot_md_tables(f, plots):
    for name, plot in plots.items():
        f.write(f"### {name}\n{plot.to_markdown()}\n")


def plot_pd_tables(f, plots):
    for name, plot in plots.items():
        f.write(f"### {name}\n```pd\n{plot.to_string()}\n```\n")
    f.write("\n")


def plot_envs(env_filenames, f):
    f.write("Green = Start pos\n\n")
    f.write("Red = Target pos\n\n")
    for i, filename in enumerate(env_filenames):
        f.write(f"![Environment {i}]({filename[len('../'):]})\n")
    f.write("\n")


def plot_csv(df, f, csv_out):
    f.write(f"```csv\n{df.to_csv()}\n```\n")
    with open(csv_out, 'w') as f2:
        # Rename the index to 'run_n'
        df.index.name = 'run_n'
        df = df.rename(columns={'agent_type': 'algorithm_name', 'env': 'env_n', 'explored': 'estates_n'})
        df['solution_found'] = df['performance'] == 1.0
        df = df.drop(columns=['performance', 'used_time'])
        df = df.sort_values(by=['algorithm_name', 'env_n'])
        df = df.reset_index()
        df = df[['algorithm_name', 'env_n', 'estates_n', 'solution_found']]
        # Sort by algorithm name, then by estate_n
        f2.write(df.to_csv())


def simulate_agent_env(agent: DiscreteAgent, env: DiscreteEnvironment):
    observation = env.initial_state()
    while not env.is_done():
        action = agent.get_action(observation)
        observation = env.process_action(action)
    return env.get_performance(), env.get_restriction_stats()
