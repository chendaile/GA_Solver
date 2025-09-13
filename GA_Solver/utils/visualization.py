import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


class Plotter:
    @staticmethod
    def plot_convergence(fitness_history: List[float], title: str = "Convergence Curve"):
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_population_diversity(population_stats: dict):
        generations = list(population_stats.keys())
        diversity = [population_stats[gen].get('diversity', 0) for gen in generations]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, diversity)
        plt.title("Population Diversity")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.grid(True)
        plt.show()