from abc import abstractmethod, ABC

import numpy as np
from numba import jit
from numba.typed import List
import tqdm, multiprocessing

from eight_queens_discrete_env import EightQueensEnvironment


class BaseLocalSearchAgent(ABC):
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def compute_directions(env_size_x, env_size_y):
        b = [0] * env_size_x
        out = [b]
        for j in range(env_size_y):
            base = b.copy()
            base[j] = 1
            out.append(base)
            base[j] = -1
            out.append(base)
        return out

    @staticmethod
    def compute_score(args):
        direction, env, config = args
        return env.score_config(BaseLocalSearchAgent.element_sum(config, direction))

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def element_sum(a, b):
        return [a[i] + b[i] for i in range(len(a))]

    @staticmethod
    def compute_n_directions(env, n):
        directions = {tuple(x) for x in BaseLocalSearchAgent.compute_directions(env.size[0], env.size[1])}
        base_dir = [*directions.copy()]
        for i in range(n):
            directions = {tuple(BaseLocalSearchAgent.element_sum(dirx, bdir)) for dirx in directions for bdir in
                          base_dir}
        return [*directions]

    def solve(self, env: EightQueensEnvironment, lookahead: int = 1):
        """Solves the environment using hill climbing, minimise score"""
        directions = BaseLocalSearchAgent.compute_n_directions(env, lookahead)
        # print(f"{len(directions)=}")
        configuration = np.random.randint(0, env.size[0], size=env.size[1:]).tolist()
        best_config = configuration
        best_score = env.score_config(configuration)
        # print(f"{best_config=}, {best_score=}")
        i = 0
        while True:
            old_best_score = best_score
            old_best_config = best_config
            best_config, best_score = self.get_best_single_step_st(configuration, directions, env)
            if self.should_stop(best_config, best_score, old_best_config, old_best_score, i):
                return old_best_config
            i += 1

    @abstractmethod
    def should_stop(self, best_config, best_score, old_best_config, old_best_score, i):
        pass

    @abstractmethod
    def get_best_single_step_st(self, configuration, directions, env):
        pass

    @abstractmethod
    def get_best_single_step_mt(self, configuration, directions, env):
        pass


class HillClimbingAgent(BaseLocalSearchAgent):
    def should_stop(self, best_config, best_score, old_best_config, old_best_score, i):
        return best_score >= old_best_score and np.all(best_config == old_best_config) or i > 10000

    def get_best_single_step_st(self, configuration, directions, env):
        best_score = 9999999999999999
        best_config = None
        for k, direction in enumerate(directions):
            new_config = BaseLocalSearchAgent.element_sum(configuration, direction)
            new_score = env.score_config(new_config)
            if best_score is None or new_score < best_score:
                best_score = new_score
                best_config = new_config
        return best_config, best_score

    def get_best_single_step_mt(self, configuration, directions, env):
        with multiprocessing.Pool() as pool:
            results = list(tqdm.tqdm(
                pool.imap(
                    BaseLocalSearchAgent.compute_score,
                    [(d, env, configuration) for d in directions]),
                total=len(directions)
            ))
            best_score = min(results)
            best_config = BaseLocalSearchAgent.element_sum(configuration, directions[np.argmin(results)])
        return best_config, best_score


class SimulatedAnnealingAgent(BaseLocalSearchAgent):

    def should_stop(self, best_config, best_score, old_best_config, old_best_score, i):
        return i > 1000 or np.all(best_config == old_best_config) or best_score == 0

    def get_best_single_step_mt(self, configuration, directions, env):
        return self.get_best_single_step_st(configuration, directions, env)

    def __init__(self, t=100, dt=0.999):
        self.t = t
        self.dt = dt

    def apply_temperature(self):
        self.t *= self.dt

    def get_best_single_step_st(self, configuration, directions, env):
        current_score = env.score_config(configuration)
        random_direction = directions[np.random.randint(0, len(directions))]
        new_config = BaseLocalSearchAgent.element_sum(configuration, random_direction)
        new_score = env.score_config(new_config)
        self.apply_temperature()
        if new_score < current_score:
            return new_config, new_score
        elif np.random.random() < np.exp((current_score - new_score) / self.t):
            return new_config, new_score
        else:
            return configuration, current_score


class GeneticAlgorithmAgent:
    def __init__(self, population_size=100, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.populate_function = GeneticAlgorithmAgent.generate_permutation_population
        self.mutation_function = GeneticAlgorithmAgent.mutate2
        self.crossover_function = GeneticAlgorithmAgent.crossover2

    def solve(self, env: EightQueensEnvironment, depth=1):
        population = self.populate_function(self.population_size, env)
        best_agent = None
        best_score = 9999999999999999
        for i in range(self.generations):
            print("Generation " + str(i) + " best score: " + str(best_score))
            population, best_agent, best_score = self.evolve_population(population, env, best_agent, best_score)
            if best_score == 0:
                return best_agent
        return best_agent

    @staticmethod
    def generate_population(population_size, env: EightQueensEnvironment):
        return [np.random.randint(0, env.size[0], size=env.size[1:]).tolist() for _ in range(population_size)]

    @staticmethod
    def generate_permutation_population(population_size, env: EightQueensEnvironment):
        return [np.random.permutation(env.size[0]).tolist() for _ in range(population_size)]

    def evolve_population(self, population, env, old_best_agent=None, old_best_score=9999999):
        scores = self.score_configs_st(env, population)
        scores = np.array(scores)
        new_best_score = np.min(scores)
        new_best_agent = population[np.argmin(scores)]
        scores = np.exp(-scores)
        scores /= np.sum(scores)
        new_population = []
        for _ in range(self.population_size):
            parent1 = population[np.random.choice(len(population), p=scores)]
            parent2 = population[np.random.choice(len(population), p=scores)]
            child = self.crossover_function(parent1, parent2)
            if np.random.random() < self.mutation_rate:
                child = self.mutation_function(child, env.size[0])
            new_population.append(child)
        if old_best_score < new_best_score:
            new_best_agent = old_best_agent
            new_best_score = old_best_score
        return new_population, new_best_agent, new_best_score

    def score_configs_st(self, env, population):
        return [env.score_config(x) for x in population]

    def score_configs_mt(self, env, population):
        with multiprocessing.Pool() as pool:
            # results = list(tqdm.tqdm(pool.imap(env.score_config, population), total=len(population)))
            results = list(pool.imap(env.score_config, population))
        return results

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def crossover(parent1, parent2):
        return [parent1[i] if np.random.random() < 0.5 else parent2[i] for i in range(len(parent1))]

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def mutate(child, env_size_x):
        child[np.random.randint(0, len(child))] = np.random.randint(0, env_size_x)
        return child

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def mutate2(child, env_size_x):
        """Swap two random elements"""
        i1 = np.random.randint(0, len(child))
        i2 = np.random.randint(0, len(child))
        child[i1], child[i2] = child[i2], child[i1]
        return child

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def crossover2(parent1, parent2):
        """Good for permutations"""
        i1 = np.random.randint(0, len(parent1))
        i2 = np.random.randint(0, len(parent1))
        if i1 > i2:
            i1, i2 = i2, i1
        child = parent1[i1:i2]
        for i in range(len(parent2)):
            if parent2[i] not in child:
                child.append(parent2[i])
        return child

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def crossover3(parent1, parent2):
        i1 = np.random.randint(0, len(parent1))
        i2 = np.random.randint(0, len(parent1))
        if i1 > i2:
            i1, i2 = i2, i1
        child1 = parent1[i1:i2]
        child2 = parent2[i1:i2]
        for i in range(len(parent2)):
            if parent2[i] not in child1:
                child1.append(parent2[i])
            if parent1[i] not in child2:
                child2.append(parent1[i])
        return child1, child2
