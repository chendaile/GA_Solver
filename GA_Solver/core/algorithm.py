import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from typing import Dict, Any, Optional, Callable
from time import time

from .population import Population
from .individual import Individual
from .problem import Problem


class GeneticAlgorithm:
    DEFAULT_CONFIG = {
        "POP_SIZE": 1000,
        "MAX_GEN": 500,
        "CROSSOVER_RATE": 0.8,
        "MUTATION_RATE": 0.1,
        "N_ELITE": 10,
        "TOURNAMENT_SIZE": 3,
        "VERBOSE": False
    }
    
    def __init__(self, problem: Problem, config: Optional[Dict[str, Any]] = None):
        self.problem = problem
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.population = None
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.start_time = None
        
    def initialize_population(self):
        self.population = Population.random_init(
            self.config["POP_SIZE"],
            (self.problem.dimensions,),
            self.problem.bounds
        )
        
    def evaluate_population(self):
        with Pool(processes=mp.cpu_count()) as pool:
            fitnesses = pool.map(self.problem.fitness, [ind.genes for ind in self.population.individuals])
            
        for individual, fitness in zip(self.population.individuals, fitnesses):
            individual.fitness = fitness
            individual.evaluated = True
            
    def selection(self) -> Individual:
        tournament_indices = np.random.choice(
            len(self.population), 
            self.config["TOURNAMENT_SIZE"], 
            replace=False
        )
        tournament = [self.population[i] for i in tournament_indices]
        return max(tournament, key=lambda x: x.fitness)
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        alpha = np.random.random()
        child_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        return Individual(child_genes)
        
    def mutate(self, individual: Individual) -> Individual:
        mutated = individual.copy()
        mutation_mask = np.random.random(len(mutated.genes)) < self.config["MUTATION_RATE"]
        mutation_values = np.random.normal(0, 0.1, len(mutated.genes))
        mutated.genes[mutation_mask] += mutation_values[mutation_mask]
        
        # 边界处理
        mutated.genes = np.clip(mutated.genes, self.problem.bounds[0], self.problem.bounds[1])
        return mutated
        
    def replacement(self, offspring: Population):
        # 精英保留
        self.population.sort_by_fitness()
        elite = self.population.individuals[:self.config["N_ELITE"]]
        
        # 合并并选择最优
        combined = elite + offspring.individuals
        combined.sort(key=lambda x: x.fitness, reverse=True)
        
        self.population.individuals = combined[:self.config["POP_SIZE"]]
        self.population.size = len(self.population.individuals)
        
    def run(self) -> Individual:
        self.start_time = time()
        self.initialize_population()
        
        try:
            for gen in range(self.config["MAX_GEN"]):
                self.generation = gen
                
                # 评估
                self.evaluate_population()
                
                # 更新最优
                current_best = self.population.get_best()
                if current_best.fitness > self.best_fitness:
                    self.best_fitness = current_best.fitness
                    self.best_individual = current_best.copy()
                    
                self.fitness_history.append(self.best_fitness)
                
                if self.config["VERBOSE"]:
                    print(f"Generation {gen}: Best Fitness = {self.best_fitness:.6f}")
                    
                # 生成下一代
                offspring = []
                while len(offspring) < self.config["POP_SIZE"] - self.config["N_ELITE"]:
                    if np.random.random() < self.config["CROSSOVER_RATE"]:
                        parent1 = self.selection()
                        parent2 = self.selection()
                        child = self.crossover(parent1, parent2)
                    else:
                        child = self.selection().copy()
                        
                    child = self.mutate(child)
                    offspring.append(child)
                    
                # 替换
                self.replacement(Population(offspring))
                
        except KeyboardInterrupt:
            print("Optimization stopped by user")
            
        end_time = time()
        if self.config["VERBOSE"]:
            print(f"Optimization completed in {end_time - self.start_time:.2f} seconds")
            
        return self.best_individual