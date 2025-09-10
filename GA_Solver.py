from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import json
from time import time


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

    def UpdateConfig(self, tmp_read=None, JSON_PATH: str = None):
        if tmp_read is None:
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

    def _evaluate_position_sensitivity(self, args):
        individual, pos, perturbation_size, fitness_func = args
        original_fitness = fitness_func(individual)

        perturbed = individual.copy()
        perturbed[pos] += perturbation_size
        perturbed_fitness = fitness_func(perturbed)

        return abs(perturbed_fitness - original_fitness)

    def UpdateAttentionMask(self):
        best_individual = self.population[np.argmax(self.scores)]
        positions = [(best_individual, tuple(pos), self.CONFIG["PERTURBATION_SIZE"], self.fitness_func)
                     for pos in np.ndindex(self.solve_shape)]

        with Pool(processes=mp.cpu_count()) as pool:
            sensitivities = pool.map(
                self._evaluate_position_sensitivity, positions)

        sensitivity_matrix = np.array(sensitivities).reshape(self.solve_shape)
        self.attention_mask = sensitivity_matrix >= self.CONFIG["SENSITIVITY_THRESHOLD"]

        if self.verbose:
            print(
                f"Have updated attention mask: {self.attention_mask}")

    def _generate_individual(self, args):
        population, operation_type, indices, attention_mask = args
        if operation_type == 'crossover':
            weights = np.random.rand(len(indices))
            weights = weights / weights.sum()
            crossover_result = np.tensordot(
                weights, population[indices], axes=1)
            base_individual = population[indices[0]]
            return base_individual * (~attention_mask) + crossover_result * attention_mask
        else:
            individual = population[indices[0]].copy()
            mutation = np.random.normal(0, self.CONFIG["STANDARD_DEVIATION"],
                                        individual.shape)
            mutation = mutation * attention_mask
            individual += mutation
            return individual

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
                tasks.append((self.population, 'crossover',
                             indices, self.attention_mask))
            else:
                indices = np.random.choice(self.population.shape[0], 1)
                tasks.append((self.population, 'mutate',
                             indices, self.attention_mask))

        with Pool(processes=mp.cpu_count()) as pool:
            new_individuals = pool.map(self._generate_individual, tasks)
        new_individuals = [ind[np.newaxis, ...] if ind.ndim == len(
            self.solve_shape) else ind for ind in new_individuals]
        self.population = np.concatenate(
            [self.population] + new_individuals, axis=0)

        if self.verbose:
            print(
                f"Build {needed} individuals, total: {self.CONFIG['POP_SIZE']} individuals")

    def Optimize(self, verbose=False):
        self.verbose = verbose
        while True:
            time_start = time()
            self.generation_count += 1
            self.BuildGeneration()
            with Pool(processes=mp.cpu_count()) as pool:
                self.scores = np.array(
                    pool.map(self.fitness_func, self.population))

            sorted_indices = np.argsort(self.scores)[
                ::-1][:self.CONFIG['N_ELITE']]
            self.population = self.population[sorted_indices]
            self.scores = self.scores[sorted_indices]

            if self.generation_count % self.CONFIG["ATTENTION_UPDATE_FREQ"] == 0:
                self.UpdateAttentionMask()

            time_end = time()
            print(
                f"GEN: {self.generation_count}, Best Score: {self.scores[0]}, Time: {time_end-time_start}")
