from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import json
from time import time
import os
from datetime import datetime
import cpuinfo


class GA_Solver:
    DEFAULT_PARAMETERS = {"POP_SIZE": 2000,
                          "STANDARD_DEVIATION": 10,
                          "N_CROSSOVER": 3,
                          "CROSSOVER_MUTATE_RATE": 0.5,
                          "ATTENTION_UPDATE_FREQ": 20,
                          "PERTURBATION_SIZE": 1.0,
                          "N_ELITE": 30,
                          "MAX_GEN": 3000,
                          "STAGNATION_THRESHOLD": 50
                          }

    def __init__(self, init_matrix: np.ndarray, fitness_func=None):
        """fitness_func need to Get fitness from matrix"""
        self.CONFIG = GA_Solver.DEFAULT_PARAMETERS.copy()
        self.solve_shape = init_matrix.shape
        self.init_matrix = init_matrix
        self.population = np.stack([self.init_matrix])
        self.fitness_func = fitness_func
        self.attention_mask = np.ones(init_matrix.shape, dtype=bool)
        self.attention_efficiencies = []
        self.generation_count = 0
        self.local_optima = []
        self.best_global_score = float('-inf')
        self.best_score = float('-inf')
        self.start_time = None
        self.end_time = None
        self.stagnation_gen = 0

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

        return perturbed_fitness - original_fitness

    def UpdateAttentionMask(self):
        best_individual = self.population[np.argmax(self.scores)]
        positions = [(best_individual, tuple(pos), self.CONFIG["PERTURBATION_SIZE"], self.fitness_func)
                     for pos in np.ndindex(self.solve_shape)]

        with Pool(processes=mp.cpu_count()) as pool:
            sensitivities = pool.map(
                self._evaluate_position_sensitivity, positions)

        self.sensitivity_matrix = np.array(
            sensitivities).reshape(self.solve_shape)
        self.attention_mask = self.sensitivity_matrix >= 0
        self.attention_efficiency = 1 - np.sum(
            self.attention_mask) / np.prod(self.solve_shape)
        self.attention_efficiencies.append(self.attention_efficiency)

        if not np.any(self.attention_mask):
            current_best = self.population[np.argmax(self.scores)]
            current_score = self.best_score
            if current_score > self.best_global_score:
                self.local_optima.append((current_best.copy(), current_score))
                self.attention_mask = np.ones(self.solve_shape, dtype=bool)
                self.best_global_score = current_score
                print(f"局部最优检测! 记录解并重启全局搜索, 局部最优数量: {len(self.local_optima)}")
                print(f"新的全局最优! 分数: {current_score}")

        if self.verbose:
            print(
                f"Have updated attention mask: \n{self.attention_mask} \nsensitive matrix: {self.sensitivity_matrix} \nefficiency: {self.attention_efficiency}")

    def _generate_individual(self, args):
        population, operation_type, indices, attention_mask, std_dev = args
        if operation_type == 'crossover':
            weights = np.random.rand(len(indices))
            weights = weights / weights.sum()
            crossover_result = np.tensordot(
                weights, population[indices], axes=1)
            base_individual = population[indices[0]]
            return base_individual * (~attention_mask) + crossover_result * attention_mask
        else:
            individual = population[indices[0]].copy()
            mutation = np.random.normal(0, std_dev, individual.shape)
            return individual + mutation * attention_mask

    def BuildGeneration(self):
        needed = self.CONFIG['POP_SIZE'] - self.population.shape[0]
        if needed <= 0:
            return

        tasks = []
        for _ in range(needed):
            if self.population.shape[0] >= self.CONFIG['N_CROSSOVER'] and \
                    np.random.rand() < self.CONFIG["CROSSOVER_MUTATE_RATE"]:
                indices = np.random.choice(
                    self.population.shape[0], self.CONFIG['N_CROSSOVER'], replace=False)
                tasks.append((self.population, 'crossover',
                             indices, self.attention_mask, self.CONFIG["STANDARD_DEVIATION"]))
            else:
                indices = np.random.choice(self.population.shape[0], 1)
                tasks.append((self.population, 'mutate',
                             indices, self.attention_mask, self.CONFIG["STANDARD_DEVIATION"]))

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
        self.start_time = time()
        print("Press Ctrl+C to stop optimization at any time")
        try:
            for _ in range(self.CONFIG["MAX_GEN"]):
                time_start = time()
                self.generation_count += 1
                self.stagnation_gen += 1
                self.BuildGeneration()
                with Pool(processes=mp.cpu_count()) as pool:
                    self.scores = np.array(
                        pool.map(self.fitness_func, self.population))

                sorted_indices = np.argsort(self.scores)[
                    ::-1][:self.CONFIG['N_ELITE']]
                self.population = self.population[sorted_indices]
                self.scores = self.scores[sorted_indices]
                if self.scores[0] > self.best_score:
                    self.stagnation_gen = 0
                self.best_score = self.scores[0]

                if self.generation_count % self.CONFIG["ATTENTION_UPDATE_FREQ"] == 0:
                    self.UpdateAttentionMask()
                if self.stagnation_gen > self.CONFIG["STAGNATION_THRESHOLD"]:
                    self.stagnation_gen = 0
                    self.CONFIG["STANDARD_DEVIATION"] *= 0.7
                    print(
                        f"Encounter stagnation, STANDARD_DEVIATION reduce to: {self.CONFIG['STANDARD_DEVIATION']}")

                time_end = time()
                print(
                    f"GEN: {self.generation_count}, Best Score: {self.best_score}, Time: {time_end-time_start}")
        except KeyboardInterrupt:
            print("\nOptimization stopped by user")
        self.end_time = time()
        self.ResultReport()

    def ResultReport(self):
        total_runtime = self.end_time - \
            self.start_time if self.end_time and self.start_time else 0
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get('brand_raw', 'Unknown CPU')

        report = []
        report.append(f"\n{'='*60}")
        report.append("Genetic Algorithm Optimization Report")
        report.append(f"{'='*60}")
        report.append(f"CPU: {cpu_name}")
        report.append(f"Total Runtime: {total_runtime:.2f} seconds")
        report.append(
            f"Total Generations: {self.generation_count}")
        report.append(f"Best Fitness: {self.best_global_score:.6f}")
        report.append(f"Local Optima Found: {len(self.local_optima)}")
        report.append(
            f"Active Attention Positions: {np.sum(self.attention_mask)}")
        report.append(f"\nFinal Attention Mask:\n{self.attention_mask}")
        report.append(f"\nFinal Sensitivity:\n{self.sensitivity_matrix}")
        report.append(
            f"\nAverage attention efficiency:\n{np.average(self.attention_efficiencies)}")
        report.append(f"\nConfiguration Parameters:")
        for key, value in self.CONFIG.items():
            report.append(f"  {key}: {value}")
        report.append(f"\nBest Individual:\n{self.population[0]}")
        if self.local_optima:
            report.append(f"\nLocal Optima History:")
            for i, (_, score) in enumerate(self.local_optima):
                report.append(f"  Local Optimum {i+1}: Score {score:.6f}")
        report.append(f"{'='*60}")

        for line in report:
            print(line)

        os.makedirs("log", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"log/ga_report_{timestamp}.txt"
        with open(log_file, 'w') as f:
            f.write("\n".join(report))
        print(f"Report saved to {log_file}")
