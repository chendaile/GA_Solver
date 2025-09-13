import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional


class Individual:
    def __init__(self, genes: np.ndarray, fitness: Optional[float] = None):
        self.genes = genes.copy()
        self.fitness = fitness
        self.evaluated = fitness is not None
        
    def copy(self):
        return Individual(self.genes, self.fitness)
        
    def __len__(self):
        return len(self.genes)
        
    def __getitem__(self, key):
        return self.genes[key]
        
    def __setitem__(self, key, value):
        self.genes[key] = value
        self.evaluated = False
        self.fitness = None