# GA_Solver

Dynamic attention-based genetic algorithm optimizer with multiprocessing support.

## Quick Start

```python
from src.GA_Solver import GA_Solver
import numpy as np

def fitness_func(matrix):
    return -np.sum(matrix**2)  # Minimize sphere function

init_matrix = np.random.rand(5, 5)
solver = GA_Solver(init_matrix, fitness_func)
solver.Optimize()
```

## Algorithm Features

**Dynamic Attention Mechanism**: Automatically identifies important solution regions through sensitivity analysis, focusing computational resources on promising areas while ignoring irrelevant dimensions.

**Adaptive Parameters**: Real-time adjustment of mutation strength based on optimization progress. Standard deviation decreases during stagnation periods to enable fine-tuning.

**Local Optima Detection**: Monitors convergence patterns and triggers global search restart when local optima are detected, maintaining solution history.

**Multiprocessing Support**: Full CPU utilization through parallel fitness evaluation, individual generation, and sensitivity computation.

**Performance Tracking**: Comprehensive logging with CPU information, runtime statistics, attention efficiency metrics, and optimization reports saved to log files.
