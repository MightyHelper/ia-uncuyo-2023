import numpy as np

from eight_queens_discrete_env import EightQueensEnvironment


class HillClimbingAgent:
    def solve(self, env: EightQueensEnvironment, lookahead: int = 1):
        """Solves the environment using hill climbing, minimise score"""
        directions = self.compute_n_directions(env, lookahead)
        print(f"{directions.shape=}")
        configuration = np.random.randint(0, env.size[0], size=env.size[1:])
        best_config = configuration
        best_score = env.score_config(configuration)
        print(f"{best_config=}, {best_score=}")
        for i in range(1000):
            old_best_score = best_score
            old_best_config = best_config
            best_config, best_score = self.get_best_single_step2(configuration, directions, env)
            if best_score >= old_best_score and np.all(best_config == old_best_config):
                return old_best_config

    def get_best_single_step(self, best_config, best_score, configuration, directions, env):
        for k, direction in enumerate(directions):
            new_config = configuration + direction
            new_score = env.score_config(new_config)
            if best_score is None or new_score < best_score:
                best_score = new_score
                best_config = new_config
        return best_config, best_score

    def get_best_single_step2(self, configuration, directions, env):
        import tqdm, multiprocessing
        with multiprocessing.Pool() as pool:
            results = list(tqdm.tqdm(pool.imap(self.compute_score, [(d, env, configuration) for d in directions]), total=len(directions)))
            best_score = min(results)
            best_config = configuration + directions[np.argmin(results)]
        return best_config, best_score

    def compute_n_directions(self, env, n):
        directions = self.compute_directions(env)
        base_dir = self.compute_directions(env)
        for i in range(n):
            directions = np.unique(np.concatenate([dirx + base_dir for dirx in directions]), axis=0)
        return directions

    def compute_directions(self, env):
        return np.concatenate([
            np.eye(env.size[0], dtype=int),
            -np.eye(env.size[0], dtype=int),
            np.zeros((1, env.size[0]), dtype=int)]
        )

    def compute_score(self, args):
        direction, env, config = args
        return env.score_config(config + direction)
