import numpy as np
from typing import List, Callable, Optional
from .individual import Individual


class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals
        self.size = len(individuals)
        
    @classmethod
    def random_init(cls, size: int, dimensions: tuple, bounds: tuple = (-10, 10)):
        individuals = []
        for _ in range(size):
            genes = np.random.uniform(bounds[0], bounds[1], dimensions)
            individuals.append(Individual(genes))
        return cls(individuals)
        
    def evaluate(self, fitness_func: Callable):
        for individual in self.individuals:
            if not individual.evaluated:
                individual.fitness = fitness_func(individual.genes)
                individual.evaluated = True
                
    def get_best(self) -> Individual:
        return max(self.individuals, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        
    def get_worst(self) -> Individual:
        return min(self.individuals, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        
    def get_fitnesses(self) -> np.ndarray:
        return np.array([ind.fitness for ind in self.individuals if ind.fitness is not None])
        
    def sort_by_fitness(self, reverse: bool = True):
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=reverse)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, key):
        return self.individuals[key]
        
    def __setitem__(self, key, value):
        self.individuals[key] = value