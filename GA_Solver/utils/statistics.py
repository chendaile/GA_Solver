import numpy as np
from typing import List, Dict, Any


class Statistics:
    def __init__(self):
        self.data = {}
        
    def record(self, generation: int, fitness: float, population_stats: Dict[str, float] = None):
        if generation not in self.data:
            self.data[generation] = {}
            
        self.data[generation]['best_fitness'] = fitness
        
        if population_stats:
            self.data[generation].update(population_stats)
            
    def get_convergence_curve(self) -> List[float]:
        return [self.data[gen]['best_fitness'] for gen in sorted(self.data.keys())]
        
    def get_statistics_summary(self) -> Dict[str, Any]:
        if not self.data:
            return {}
            
        convergence = self.get_convergence_curve()
        
        return {
            'final_fitness': convergence[-1] if convergence else None,
            'initial_fitness': convergence[0] if convergence else None,
            'improvement': convergence[-1] - convergence[0] if len(convergence) >= 2 else 0,
            'generations': len(convergence),
            'convergence_curve': convergence
        }