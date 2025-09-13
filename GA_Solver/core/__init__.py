"""
核心模块 - 遗传算法的基础组件
"""

from .algorithm import GeneticAlgorithm
from .individual import Individual
from .population import Population
from .problem import Problem

__all__ = ['GeneticAlgorithm', 'Individual', 'Population', 'Problem']