from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import json


class GA_Solver:
    DEFAULT_PARAMETERS = {"POP_SIZE": 100,
                          "MAX_GEN": 100,
                          "STANDARD_DEVIATION": 10,
                          "N_CROSSOVER": 3,
                          "N_ELITE": 10,
                          "CROSSOVER_MUTATE_RATE": 0.5,
                          "SENSITIVITY_THRESHOLD": 0.01,
                          "ATTENTION_UPDATE_FREQ": 10,
                          "PERTURBATION_SIZE": 1.0
                          }

    def __init__(self, init_matrix: np.ndarray, fitness_func=None):
        """fitness_func need to Get fitness from matrix"""
        self.CONFIG = GA_Solver.DEFAULT_PARAMETERS
        self.solve_shape = init_matrix.shape
        self.init_matrix = init_matrix
        self.population = np.stack([self.init_matrix])
        self.fitness_func = fitness_func
        self.attention_mask = np.ones(init_matrix.shape, dtype=bool)
        self.generation_count = 0

    def UpdateConfig(self, JSON_PATH: str):
        try:
            with open(JSON_PATH, 'r') as f:
                tmp_read = json.load(f)
        except Exception as e:
            print(f"Wrong reading json: {e}")

        for key, value in tmp_read.items():
            if key not in self.CONFIG.keys():
                print(f"Wrong parameter {key}")
            self.CONFIG[key] = value
            print(f"Change {key}'s value to {value}")

    def Get_MutateIndividual(self, individual: np.ndarray):
        tmp_individual = individual.copy()
        tmp_individual += np.random.normal(0,
                                           self.CONFIG["STANDARD_DEVIATION"])
        return tmp_individual

    def Get_CrossoverIndividuals(self, individuals):
        weights = np.random.rand(len(individuals))
        weights = weights / weights.sum()
        return np.tensordot(weights, individuals, axes=1)

    def _generate_individual(self, args):
        population, operation_type, indices = args
        if operation_type == 'crossover':
            return self.Get_CrossoverIndividuals(population[indices])
        else:
            return self.Get_MutateIndividual(population[indices[0]])

    def BuildGeneration(self):
        needed = self.CONFIG['POP_SIZE'] - self.population.shape[0]
        if needed <= 0:
            return

        tasks = []
        for _ in range(needed):
            if self.population.shape[0] >= self.CONFIG['N_CROSSOVER'] and \
                    np.random.rand() > self.CONFIG["CROSSOVER_MUTATE_RATE"]:
                indices = np.random.choice(
                    self.population.shape[0], self.CONFIG['N_CROSSOVER'], replace=False)
                tasks.append((self.population, 'crossover', indices))
            else:
                indices = np.random.choice(self.population.shape[0], 1)
                tasks.append((self.population, 'mutate', indices))

        with Pool(processes=mp.cpu_count()) as pool:
            new_individuals = pool.map(self._generate_individual, tasks)
        self.population = np.concatenate(
            [self.population] + new_individuals, axis=0)

    def Optimize(self):
        self.BuildGeneration()
        with Pool(processes=mp.cpu_count()) as pool:
            self.scores = np.array(
                pool.map(self.fitness_func, self.population))
        sorted_indices = np.argsort(self.scores)[::-1][self.CONFIG['N_ELITE']]
        self.population = self.population[sorted_indices]
        self.scores = self.scores[sorted_indices]
