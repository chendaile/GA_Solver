"""
GA Library - 模块化遗传算法库

主要特性:
- 灵活的参数实时变化
- 专注力随时变化的优化模型  
- CPU多进程并行支持
- 模块化设计，易于扩展
"""

__version__ = "1.0.0"
__author__ = "GA_Solver Project"

from .core.algorithm import GeneticAlgorithm
from .core.individual import Individual
from .core.population import Population
from .core.problem import Problem

from .operators.selection import (
    RouletteWheelSelection,
    TournamentSelection, 
    RankSelection
)
from .operators.crossover import (
    ArithmeticCrossover,
    BlendCrossover,
    SimulatedBinaryCrossover
)
from .operators.mutation import (
    GaussianMutation,
    PolynomialMutation,
    UniformMutation
)

from .problems.benchmark import BenchmarkProblems
from .utils.statistics import Statistics

__all__ = [
    'GeneticAlgorithm',
    'Individual', 
    'Population',
    'Problem',
    'RouletteWheelSelection',
    'TournamentSelection',
    'RankSelection',
    'ArithmeticCrossover',
    'BlendCrossover', 
    'SimulatedBinaryCrossover',
    'GaussianMutation',
    'PolynomialMutation',
    'UniformMutation',
    'BenchmarkProblems',
    'Statistics'
]