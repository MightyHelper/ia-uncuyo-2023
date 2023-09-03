import test_runner as test

if __name__ == '__main__':
    test.main('../tp4-reporte.md', '../informada-results.csv', ['random', 'dfs', 'bfs', 'dijkstra', 'ldfs-00.5', 'ldfs-01.0', 'ldfs-02.0', 'ldfs-04.0', 'ldfs-16.0', 'a*'], 4, n_envs=30)
