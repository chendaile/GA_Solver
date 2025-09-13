from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple


class Problem(ABC):
    def __init__(self, dimensions: int, bounds: Tuple[float, float] = (-10, 10)):
        self.dimensions = dimensions
        self.bounds = bounds
        
    @abstractmethod
    def fitness(self, genes: np.ndarray) -> float:
        pass
        
    def constraints(self, genes: np.ndarray) -> List[float]:
        return []
        
    def is_feasible(self, genes: np.ndarray) -> bool:
        violations = self.constraints(genes)
        return all(v <= 0 for v in violations)
        
    def evaluate_with_penalty(self, genes: np.ndarray, penalty_factor: float = 1000) -> float:
        fitness = self.fitness(genes)
        violations = self.constraints(genes)
        
        if violations:
            total_violation = sum(max(0, v) for v in violations)
            if total_violation > 0:
                return fitness - penalty_factor * total_violation
                
        return fitness