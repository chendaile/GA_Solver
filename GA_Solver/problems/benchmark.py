import numpy as np
from ..core.problem import Problem


class SphereFunction(Problem):
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-5, 5))
        
    def fitness(self, genes: np.ndarray) -> float:
        return -np.sum(genes**2)


class RosenbrockFunction(Problem):
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-2, 2))
        
    def fitness(self, genes: np.ndarray) -> float:
        x = genes.flatten()
        return -sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class AckleyFunction(Problem):
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-5, 5))
        
    def fitness(self, genes: np.ndarray) -> float:
        x = genes.flatten()
        n = len(x)
        a, b, c = 20, 0.2, 2*np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
        term2 = -np.exp(np.sum(np.cos(c * x)) / n)
        return -(term1 + term2 + a + np.e)


class RastriginFunction(Problem):
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-5, 5))
        
    def fitness(self, genes: np.ndarray) -> float:
        x = genes.flatten()
        A = 10
        n = len(x)
        return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


class BenchmarkProblems:
    @staticmethod
    def get_problem(name: str, dimensions: int = 10) -> Problem:
        problems = {
            'sphere': SphereFunction,
            'rosenbrock': RosenbrockFunction,
            'ackley': AckleyFunction,
            'rastrigin': RastriginFunction
        }
        
        if name not in problems:
            raise ValueError(f"Unknown problem: {name}. Available: {list(problems.keys())}")
            
        return problems[name](dimensions)